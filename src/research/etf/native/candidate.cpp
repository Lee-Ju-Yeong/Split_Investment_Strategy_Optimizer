// Exact-rational M3 candidate kernel. No portfolio state or market constants.
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Integer = boost::multiprecision::cpp_int;
using Ratio = boost::rational<Integer>;

Integer floor_value(const Ratio& value) {
    Integer quotient = value.numerator() / value.denominator();
    if (value.numerator() < 0 && value.numerator() % value.denominator() != 0) --quotient;
    return quotient;
}
Integer ceil_value(const Ratio& value) { return -floor_value(-value); }

Ratio power_of_ten(int exponent) {
    Integer value = 1, base = 10;
    for (int power = std::abs(exponent); power; power /= 2) {
        if (power % 2 != 0) value *= base;
        base *= base;
    }
    return exponent >= 0 ? Ratio(value) : Ratio(Integer(1), value);
}
int adjusted_exponent(const Ratio& positive) {
    int exponent = static_cast<int>(positive.numerator().str().size())
                   - static_cast<int>(positive.denominator().str().size());
    if (positive < power_of_ten(exponent)) --exponent;
    return exponent;
}
bool small_decimal(const Ratio& value) {
    static const Integer bound("1000000000000000000");
    if (value.numerator() <= -bound || value.numerator() >= bound || value.denominator() >= bound) return false;
    auto denominator = value.denominator().convert_to<unsigned long long>();
    while (denominator % 2 == 0) denominator /= 2;
    while (denominator % 5 == 0) denominator /= 5;
    return denominator == 1;
}
bool round_away(const Ratio& remainder, const Integer& integer, int mode, bool negative) {
    if (remainder == Ratio(0)) return false;
    if (mode == 1) return true;
    if (mode == 2) return false;
    if (mode == 3) return !negative;
    if (mode == 4) return negative;
    if (mode == 5) return remainder >= Ratio(1, 2);
    if (mode == 6) return remainder > Ratio(1, 2);
    if (mode == 7) return integer % 10 == 0 || integer % 10 == 5;
    return remainder > Ratio(1, 2) || (remainder == Ratio(1, 2) && integer % 2 != 0);
}
Ratio decimal80(const Ratio& value, int mode) {
    // With numerator/denominator below 1e18, a finite decimal has <80 digits.
    if (value == Ratio(0) || small_decimal(value)) return value;
    Ratio positive = value < Ratio(0) ? -value : value;
    Ratio unit = power_of_ten(adjusted_exponent(positive) - 79);
    Ratio scaled = positive / unit;
    Integer integer = floor_value(scaled);
    Ratio remainder = scaled - Ratio(integer);
    if (round_away(remainder, integer, mode, value < Ratio(0))) ++integer;
    return Ratio(value < Ratio(0) ? -integer : integer) * unit;
}
Ratio previous_decimal(const Ratio& positive) {
    int exponent = adjusted_exponent(positive);
    Ratio unit = power_of_ten(exponent - 79);
    Ratio down = Ratio(floor_value(positive / unit)) * unit;
    if (down < positive) return down;
    return positive - power_of_ten(exponent - (positive == power_of_ten(exponent) ? 80 : 79));
}

template <class T> T read(std::istream& input) {
    T value;
    if (!(input >> value)) throw std::invalid_argument("invalid kernel payload");
    return value;
}
Ratio read_ratio(std::istream& input) {
    auto value = read<std::string>(input);
    auto slash = value.find('/');
    if (slash == std::string::npos) return Ratio(Integer(value));
    return Ratio(Integer(value.substr(0, slash)), Integer(value.substr(slash + 1)));
}
int read_count(std::istream& input) {
    int count = read<int>(input);
    if (count < 0 || count > 100000) throw std::invalid_argument("invalid kernel count");
    return count;
}

struct Band { Ratio below, tick; bool last; };
struct Rule { int level; Ratio budget, drop; };
struct Holding { int level, priority; Integer quantity; Ratio entry, fee, take; };
struct Order { bool sell; int level; Integer quantity; Ratio price; };
struct Node { Ratio at, price; };
struct Signal { Ratio at, price; bool after; };
struct Fill { Order order; Integer micros; std::string reason; };
using MaybeOrder = std::optional<Order>;
using MaybeRatio = std::optional<Ratio>;
using MaybeFill = std::optional<Fill>;

struct Context {
    std::array<Node, 4> nodes;
    std::vector<Band> bands;
    std::vector<Rule> rules;
    std::vector<Holding> held;
    Ratio buy_fee, sell_fee, tax;
    Integer duration;
    Integer penetration, slippage;
    int rounding;
    bool manual;
};

Context parse(std::istream& input) {
    Context context;
    context.duration = read<Integer>(input);
    context.penetration = read<Integer>(input); context.slippage = read<Integer>(input);
    context.manual = read<int>(input) != 0;
    context.rounding = read<int>(input);
    context.buy_fee = read_ratio(input); context.sell_fee = read_ratio(input); context.tax = read_ratio(input);
    for (int index = 0; index < 4; ++index) context.nodes[index] = {Ratio(index, 3), read_ratio(input)};
    for (int count = read_count(input); count > 0; --count) {
        bool last = read<int>(input) != 0;
        context.bands.push_back({read_ratio(input), read_ratio(input), last});
    }
    for (int count = read_count(input); count > 0; --count)
        context.rules.push_back({read<int>(input), read_ratio(input), read_ratio(input)});
    for (int count = read_count(input); count > 0; --count)
        context.held.push_back({read<int>(input), read<int>(input), read<Integer>(input),
                                read_ratio(input), read_ratio(input), read_ratio(input)});
    return context;
}

Ratio tick(const Context& context, const Ratio& price, bool below = false) {
    for (const auto& band : context.bands)
        if (band.last || price < band.below || (below && price == band.below)) return band.tick;
    throw std::invalid_argument("price outside tick table");
}
MaybeRatio step(const Context& context, Ratio price, const Integer& steps) {
    Integer count = steps > 0 ? steps : -steps;
    for (Integer index = 0; index < count; ++index) {
        // For short decimal prices/bands, next_minus can only cross an equal band edge.
        bool short_price = small_decimal(price) && std::all_of(context.bands.begin(), context.bands.end(),
            [](const Band& band) { return band.last || small_decimal(band.below); });
        Ratio probe = steps > 0 || short_price ? price : std::max(previous_decimal(price), power_of_ten(-100));
        price = decimal80(price + (steps > 0 ? tick(context, price) : -tick(context, probe, short_price)), context.rounding);
        if (price <= Ratio(0)) return std::nullopt;
    }
    return price;
}
Ratio round_approximate(const Context& context, const Ratio& mark, bool sell) {
    Ratio approximate = decimal80(mark, context.rounding);
    Ratio unit = tick(context, approximate);
    Ratio quotient = decimal80(approximate / unit, context.rounding);
    Ratio price = decimal80(Ratio(sell ? ceil_value(quotient) : floor_value(quotient)) * unit, context.rounding);
    if (price <= Ratio(0)) throw std::invalid_argument("rounded buy price is zero");
    return price;
}
Ratio rounded(const Context& context, const Ratio& mark, bool sell) {
    Ratio price = round_approximate(context, mark, sell);
    int direction = sell ? 1 : -1;
    while (sell ? price < mark : price > mark) {
        auto next = step(context, price, direction);
        if (!next) throw std::invalid_argument("rounded buy price is zero");
        price = *next;
    }
    auto previous = step(context, price, -direction);
    while (previous && (sell ? *previous >= mark : *previous <= mark)) {
        price = *previous; previous = step(context, price, -direction);
    }
    return price;
}
MaybeRatio limit(const Context& context, const Ratio& mark, bool sell) {
    return step(context, rounded(context, mark, sell), sell ? -context.slippage : context.slippage);
}

bool drop_met(const Holding& previous, const Ratio& mark, const Rule& rule) {
    return mark <= previous.entry * (Ratio(1) - rule.drop);
}
bool take_met(const Context& context, const Holding& holding, const Ratio& mark) {
    Ratio basis = holding.entry * Ratio(holding.quantity);
    Ratio profit = mark * Ratio(holding.quantity) * (Ratio(1) - context.sell_fee - context.tax)
                   - basis - holding.fee;
    Ratio units = profit / basis * Ratio(10000);
    Integer units_rounded = floor_value((units < Ratio(0) ? -units : units) + Ratio(1, 2));
    if (units < Ratio(0)) units_rounded = -units_rounded;
    return Ratio(units_rounded, Integer(10000)) >= holding.take;
}
Integer quantity(const Context& context, const Rule& rule, const Ratio& price) {
    Integer lower = 0, upper = floor_value(rule.budget / price);
    while (lower < upper) {
        Integer middle = (lower + upper + 1) / 2;
        Ratio gross = decimal80(price * Ratio(middle), context.rounding);
        Ratio fee = decimal80(gross * context.buy_fee, context.rounding);
        if (decimal80(gross + fee, context.rounding) <= rule.budget) lower = middle;
        else upper = middle - 1;
    }
    return lower;
}
const Holding* held_at(const Context& context, int level) {
    for (const auto& holding : context.held) if (holding.level == level) return &holding;
    return nullptr;
}
const Rule* buy_rule(const Context& context) {
    for (const auto& rule : context.rules)
        if (rule.level > 1 && held_at(context, rule.level) == nullptr) return &rule;
    return nullptr;
}
MaybeOrder sale(const Context& context, const Holding& holding, const Ratio& mark) {
    auto price = limit(context, mark, true);
    if (price && take_met(context, holding, mark) && take_met(context, holding, *price))
        return Order{true, holding.level, holding.quantity, *price};
    return std::nullopt;
}
MaybeOrder purchase(const Context& context, const Rule& rule, const Holding& previous, const Ratio& mark) {
    auto price = limit(context, mark, false);
    if (!price || !drop_met(previous, mark, rule) || !drop_met(previous, *price, rule)) return std::nullopt;
    Integer count = quantity(context, rule, *price);
    if (count > 0) return Order{false, rule.level, count, *price};
    return std::nullopt;
}
MaybeOrder order_at(const Context& context, const Ratio& mark) {
    MaybeOrder selected;
    int priority = -1;
    for (const auto& holding : context.held) {
        auto order = sale(context, holding, mark);
        if (order && (priority < 0 || holding.priority < priority)) { selected = order; priority = holding.priority; }
    }
    if (selected) return selected;
    const Rule* rule = buy_rule(context);
    const Holding* previous = rule ? held_at(context, rule->level - 1) : nullptr;
    return previous ? purchase(context, *rule, *previous, mark) : std::nullopt;
}

MaybeRatio first_fill(const Context& context, const Order& order, const Ratio& after) {
    auto target = step(context, order.price, order.sell ? context.penetration : -context.penetration);
    if (!target) return std::nullopt;
    for (int index = 0; index < 3; ++index) {
        auto start = context.nodes[index], end = context.nodes[index + 1];
        if (end.at < after) continue;
        Ratio current = std::max(start.at, after);
        Ratio value = start.price + (end.price - start.price) * (current - start.at) / (end.at - start.at);
        if (order.sell ? value >= *target : value <= *target) return current;
        if (start.price != end.price && std::min(value, end.price) <= *target && *target <= std::max(value, end.price))
            return start.at + (end.at - start.at) * (*target - start.price) / (end.price - start.price);
    }
    return std::nullopt;
}
MaybeFill fill_submitted(const Context& context, const Order& order, const Signal& signal) {
    auto when = first_fill(context, order, signal.at);
    if (when && *when == signal.at && signal.after) {
        Ratio after(floor_value(signal.at * Ratio(context.duration)) + 1, context.duration);
        when = after <= Ratio(1) ? first_fill(context, order, after) : std::nullopt;
    }
    if (!when) return std::nullopt;
    return Fill{order, ceil_value(*when * Ratio(context.duration)),
                order.sell ? "sell_condition_touch" : "buy_condition_touch"};
}
MaybeFill opening(const Context& context, const Order& order) {
    Ratio mark = context.nodes[0].price;
    auto price = step(context, mark, order.sell ? -context.slippage : context.slippage);
    if (price) {
        bool valid = order.sell ? take_met(context, *held_at(context, order.level), *price)
                               : drop_met(*held_at(context, order.level - 1), *price, *buy_rule(context));
        if (valid) {
            Integer count = order.sell ? order.quantity : quantity(context, *buy_rule(context), *price);
            if (count > 0) return Fill{{order.sell, order.level, count, *price}, 0,
                                      order.sell ? "sell_condition_gap" : "buy_condition_gap"};
        }
    }
    return fill_submitted(context, order, {Ratio(0), mark, false});
}
MaybeRatio take_boundary(const Context& context, const Holding& holding) {
    Ratio net = Ratio(1) - context.sell_fee - context.tax;
    if (net <= Ratio(0)) return std::nullopt;
    Integer units = ceil_value(holding.take * Ratio(10000));
    Ratio basis = holding.entry * Ratio(holding.quantity);
    return (basis + holding.fee + basis * (Ratio(units) - Ratio(1, 2)) / Ratio(10000))
            / (Ratio(holding.quantity) * net);
}
std::set<Ratio> boundaries(const Context& context) {
    Ratio low = context.nodes[0].price, high = low;
    for (const auto& node : context.nodes) { low = std::min(low, node.price); high = std::max(high, node.price); }
    std::set<Ratio> values;
    for (Ratio mark = round_approximate(context, low, true); mark <= high; mark = *step(context, mark, 1)) values.insert(mark);
    for (const auto& holding : context.held) {
        auto boundary = take_boundary(context, holding);
        if (boundary) values.insert(*boundary);
    }
    const Rule* rule = buy_rule(context);
    const Holding* previous = rule ? held_at(context, rule->level - 1) : nullptr;
    if (previous) values.insert(previous->entry * (Ratio(1) - rule->drop));
    return values;
}
MaybeFill intraday(const Context& context) {
    auto values = boundaries(context);
    for (int index = 0; index < 3; ++index) {
        auto start = context.nodes[index], end = context.nodes[index + 1];
        std::map<Ratio, Ratio> points{{start.at, start.price}, {end.at, end.price}};
        if (start.price != end.price) for (const auto& mark : values)
            if (std::min(start.price, end.price) < mark && mark < std::max(start.price, end.price))
                points[start.at + (end.at - start.at) * (mark - start.price) / (end.price - start.price)] = mark;
        for (auto current = points.begin(); current != points.end(); ++current) {
            auto selected = order_at(context, current->second);
            if (selected) return fill_submitted(context, *selected, {current->first, current->second, false});
            auto next = std::next(current);
            if (next == points.end() || current->second == next->second) continue;
            Ratio middle = (current->second + next->second) / Ratio(2);
            selected = order_at(context, middle);
            if (selected) return fill_submitted(context, *selected, {current->first, middle, true});
        }
    }
    return std::nullopt;
}
MaybeFill candidate(const Context& context) {
    if (context.manual && context.held.empty()) {
        if (context.rules.empty()) return std::nullopt;
        auto price = step(context, context.nodes[0].price, context.slippage);
        Integer count = price ? quantity(context, context.rules[0], *price) : Integer(0);
        if (count > 0) return Fill{{false, 1, count, *price}, 0, "manual_first_entry"};
        return std::nullopt;
    }
    auto selected = order_at(context, context.nodes[0].price);
    if (selected) return opening(context, *selected);
    return context.held.empty() ? std::nullopt : intraday(context);
}
std::string serialize(const MaybeFill& fill) {
    if (!fill) return "NONE";
    std::ostringstream output;
    const auto& order = fill->order;
    output << (order.sell ? "SELL" : "BUY") << ' ' << order.level << ' ' << order.quantity << ' '
           << order.price.numerator() << ' ' << order.price.denominator() << ' ' << fill->micros << ' ' << fill->reason;
    return output.str();
}
}  // namespace

extern "C" int etf_kernel_abi() { return 2; }
extern "C" int etf_candidate(const char* payload, char* output, std::size_t capacity) noexcept {
    try {
        std::istringstream input(payload);
        std::string result = serialize(candidate(parse(input)));
        if (result.size() + 1 > capacity) return 2;
        std::memcpy(output, result.c_str(), result.size() + 1);
        return 0;
    } catch (const std::exception& error) {
        if (capacity) { std::strncpy(output, error.what(), capacity - 1); output[capacity - 1] = '\0'; }
        return 1;
    }
}
