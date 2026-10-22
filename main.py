"""
Голландский аукцион — учебная симуляция.
Запуск: python main.py
"""

# Голландский аукцион
#
# Симуляция рынка: генерация истории цен, агенты с разными стратегиями, аукцион, распродажа лотов в течение 30 дней и сравнение прибыли.
#
# Как запускать: выполните ячейки сверху вниз. В последней секции находится главный запуск: он проводит симуляцию, строит графики и сохраняет результаты в папку auction_output.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

HISTORY_DAYS = 180       # сколько дней истории видят агенты
SELL_DAYS = 30           # сколько дней идёт распродажа лота
MIN_PRICE_SHARE = 0.1    # цена не может упасть ниже 10% от справедливой

# Параметры каждого товара:
# fair  - истинная справедливая цена (агентам неизвестна!)
# theta - скорость возврата к fair (0 < theta < 1)
# sigma - волатильность (в рублях за единицу)
PRODUCTS = {
    "роза":    {"fair": 5.0,  "theta": 0.10, "sigma": 0.20},
    "пион":    {"fair": 12.0, "theta": 0.10, "sigma": 0.50},
    "тюльпан": {"fair": 3.0,  "theta": 0.15, "sigma": 0.10},
}

# 1. Генерация рыночных цен
#
# Цена меняется по модели mean reversion: со временем она тяготеет к справедливой цене, но каждый день испытывает случайные колебания. Генератор rng передаётся в функцию, чтобы результаты можно было воспроизводить с помощью seed.


def generate_price_series(fair, theta, sigma, days, rng):
    """Строит ряд цен по модели mean reversion.

    fair  - справедливая цена, к которой тянется ряд
    theta - скорость возврата к fair
    sigma - волатильность
    days  - сколько всего цен нужно
    rng   - генератор случайных чисел (определяет воспроизводимость)
    """
    min_price = MIN_PRICE_SHARE * fair
    prices = [fair]                      # первая цена равна fair

    for day in range(1, days):
        previous_price = prices[-1]
        noise = rng.normal(0, 1)         # eps из N(0, 1)
        new_price = previous_price + theta * (fair - previous_price) + sigma * noise
        new_price = max(new_price, min_price)   # цена не уходит в минус
        prices.append(new_price)

    return np.array(prices)


class Market:
    """Хранит цены всех товаров. Единственное место, где известна истинная fair."""

    def __init__(self, products, seed):
        self.products = products
        self.rng = np.random.default_rng(seed)
        total_days = HISTORY_DAYS + SELL_DAYS
        self.prices = {}
        for name, params in products.items():
            self.prices[name] = generate_price_series(
                params["fair"], params["theta"], params["sigma"],
                total_days, self.rng,
            )

    def get_history(self, product):
        """180 дней истории: это видят агенты."""
        return self.prices[product][:HISTORY_DAYS]

    def get_future(self, product):
        """30 дней после аукциона: нужны только для распродажи."""
        return self.prices[product][HISTORY_DAYS:]

    def get_true_fair(self, product):
        """Истинная fair: только для подсчёта ошибок, агентам не передаётся."""
        return self.products[product]["fair"]

# 2. Лоты
#
# Лот хранит товар, количество, историю рыночных цен и параметры снижающейся цены на аукционе.


MIN_PRICE_SHARE_LOT = 0.5   # min_price лота = 50% от стартовой цены
STEP_SHARE = 0.01           # шаг снижения = 1% от стартовой цены

# Лоты: (товар, количество единиц)
LOTS = [
    ("роза", 100),
    ("пион", 50),
    ("тюльпан", 200),
    ("роза", 80),
]


class Lot:
    """Оптовый лот. Знает свои параметры и историю цен (но не будущее и не fair)."""

    def __init__(self, lot_id, product, quantity, history, min_price_share, step_share):
        self.lot_id = lot_id
        self.product = product
        self.quantity = quantity
        self.history = history                       # 180 дней цен за единицу
        self.current_market_price = history[-1]      # цена за единицу сегодня
        self.start_price = self.current_market_price * quantity   # цена всего лота
        self.min_price = self.start_price * min_price_share
        self.step = self.start_price * step_share

    def get_price_levels(self):
        """Возвращает список цен лота: стартовая, потом на шаг ниже, и так до минимума."""
        # сколько целых шагов помещается между стартом и минимумом
        # (+ 1e-9 защищает от погрешности дробных чисел)
        steps_count = int((self.start_price - self.min_price) / self.step + 1e-9)
        levels = []
        for step_number in range(steps_count + 1):
            price = self.start_price - step_number * self.step
            price = max(price, self.min_price)   # не ниже минимальной цены
            levels.append(price)
        return levels

    def unit_price(self, lot_price):
        """Переводит цену всего лота в цену за одну единицу."""
        return lot_price / self.quantity


def create_lots(market, lots_settings, min_price_share, step_share):
    lots = []
    for number, (product, quantity) in enumerate(lots_settings, start=1):
        lot = Lot(number, product, quantity, market.get_history(product),
                  min_price_share, step_share)
        lots.append(lot)
    return lots

# 3. Методы оценки справедливой цены
#
# Агенты не знают истинную справедливую цену. Каждый использует только доступную историю и собственный метод оценки.


class MeanEstimator:
    """Среднее цен за последние window дней."""

    def __init__(self, window):
        self.window = window
        self.name = f"mean_{window}"

    def estimate(self, history):
        recent_prices = history[-self.window:]
        return float(np.mean(recent_prices))


class MedianEstimator:
    """Медиана цен за последние window дней."""

    def __init__(self, window):
        self.window = window
        self.name = f"median_{window}"

    def estimate(self, history):
        recent_prices = history[-self.window:]
        return float(np.median(recent_prices))


class EmaEstimator:
    """Экспоненциальное скользящее среднее: свежие цены весят больше старых."""

    def __init__(self, alpha):
        self.alpha = alpha            # 0 < alpha < 1, чем больше, тем важнее свежие цены
        self.name = f"ema_{alpha}"

    def estimate(self, history):
        ema = history[0]              # начинаем с самой первой цены
        for price in history[1:]:
            ema = self.alpha * price + (1 - self.alpha) * ema
        return float(ema)

# 4. Стратегии и агенты
#
# Стратегия отвечает на вопрос «покупать ли по этой цене?», а оценщик — «какова справедливая цена?». Это две разные части поведения агента.


PRICE_TOLERANCE = 1e-9   # допуск на погрешность дробных чисел при сравнении цен


class AggressiveStrategy:
    """Берёт сразу, как только цена за единицу не выше рыночной."""

    def __init__(self):
        self.name = "Aggressive"

    def should_buy(self, lot, lot_price, fair_estimate):
        return lot.unit_price(lot_price) <= lot.current_market_price + PRICE_TOLERANCE


class CautiousStrategy:
    """Берёт, только когда цена за единицу не выше k * (своя оценка fair), k < 1."""

    def __init__(self, k):
        self.k = k
        self.name = f"Cautious(k={k})"

    def should_buy(self, lot, lot_price, fair_estimate):
        return lot.unit_price(lot_price) <= self.k * fair_estimate + PRICE_TOLERANCE


class RandomStrategy:
    """Если цена не выше рыночной, берёт с вероятностью p на каждом шаге."""

    def __init__(self, probability, rng):
        self.probability = probability
        self.rng = rng
        self.name = f"Random(p={probability})"

    def should_buy(self, lot, lot_price, fair_estimate):
        if lot.unit_price(lot_price) > lot.current_market_price + PRICE_TOLERANCE:
            return False
        return self.rng.random() < self.probability


class MovingAverageStrategy:
    """Берёт, когда цена за единицу не выше среднего рыночных цен за последние N дней."""

    def __init__(self, window):
        self.window = window
        self.name = f"MovingAverage(N={window})"

    def should_buy(self, lot, lot_price, fair_estimate):
        recent_average = float(np.mean(lot.history[-self.window:]))
        return lot.unit_price(lot_price) <= recent_average + PRICE_TOLERANCE


class WaitForDiscountStrategy:
    """Ждёт, пока цена лота упадёт на discount (например, 15%) от стартовой."""

    def __init__(self, discount):
        self.discount = discount
        self.name = f"WaitForDiscount(d={discount})"

    def should_buy(self, lot, lot_price, fair_estimate):
        return lot_price <= (1 - self.discount) * lot.start_price + PRICE_TOLERANCE


class Agent:
    """Участник аукциона: деньги + правило покупки + способ оценки fair."""

    def __init__(self, name, budget, strategy, estimator):
        self.name = name
        self.initial_budget = budget     # сколько денег было в начале
        self.budget = budget             # сколько осталось сейчас
        self.strategy = strategy
        self.estimator = estimator
        self.portfolio = []              # купленные лоты
        self.profit = 0.0                # заполним после распродажи
        self.fair_estimates = {}         # оценки fair по лотам: {lot_id: число}

    def get_fair_estimate(self, lot):
        """Оценка fair по истории лота. Считаем один раз и запоминаем."""
        if lot.lot_id not in self.fair_estimates:
            self.fair_estimates[lot.lot_id] = self.estimator.estimate(lot.history)
        return self.fair_estimates[lot.lot_id]

    def can_afford(self, price):
        return self.budget >= price

    def wants_to_buy(self, lot, lot_price):
        """Главный вопрос на каждом шаге аукциона: брать или ждать?"""
        if not self.can_afford(lot_price):
            return False
        fair_estimate = self.get_fair_estimate(lot)
        return self.strategy.should_buy(lot, lot_price, fair_estimate)

    def buy(self, lot, lot_price):
        self.budget = self.budget - lot_price
        self.portfolio.append({"lot": lot, "price_paid": lot_price})

# 5. Проведение аукциона
#
# Цена лота постепенно снижается. Первый шаг, на котором кто-либо соглашается купить, определяет победителя. Если несколько агентов согласились одновременно, победитель выбирается случайно.


class Auction:
    """Проводит торги по лотам по очереди и определяет победителей."""

    def __init__(self, lots, agents, rng):
        self.lots = lots
        self.agents = agents
        self.rng = rng            # нужен для жребия, если согласились несколько агентов

    def run_lot(self, lot):
        """Торги по одному лоту. Возвращает словарь с результатом."""
        # ready_prices[имя агента] = цена, на которой он впервые согласился (или None)
        ready_prices = {}
        for agent in self.agents:
            ready_prices[agent.name] = None

        winner = None
        price_paid = None
        winner_step = None

        for step_number, price in enumerate(lot.get_price_levels()):
            willing_agents = []     # кто согласился именно на этом шаге

            for agent in self.agents:
                if ready_prices[agent.name] is not None:
                    continue        # этот агент уже согласился раньше
                if agent.wants_to_buy(lot, price):
                    ready_prices[agent.name] = price
                    willing_agents.append(agent)

            # Победитель определяется один раз: на первом шаге, где кто-то согласился
            if winner is None and len(willing_agents) > 0:
                index = self.rng.integers(len(willing_agents))   # жребий при ничьей
                winner = willing_agents[index]
                price_paid = price
                winner_step = step_number
                winner.buy(lot, price)

        return {
            "lot": lot,
            "winner": winner,                # None, если лот не купили
            "price_paid": price_paid,
            "winner_step": winner_step,
            "ready_prices": ready_prices,
        }

    def run_all(self):
        """Проводит торги по всем лотам по очереди."""
        results = []
        for lot in self.lots:
            results.append(self.run_lot(lot))
        return results

# 6. Распродажа и расчёт результатов
#
# После покупки лот распродаётся в течение 30 дней. Количество проданных единиц меняется от дня ко дню, но в сумме равно размеру лота.


DIRICHLET_ALPHA = 1   # чем меньше, тем неравномернее продажи по дням


def make_sales_plan(quantity, days, alpha, rng):
    """Раскладывает quantity единиц по дням. Сумма всегда ровно quantity."""
    weights = rng.dirichlet([alpha] * days)            # случайные доли дней, сумма долей = 1
    units_per_day = rng.multinomial(quantity, weights) # целое число единиц на каждый день
    return units_per_day


def sell_lot(lot, price_paid, future_prices, alpha, rng):
    """Распродаёт лот по рыночным ценам следующих 30 дней."""
    units_per_day = make_sales_plan(lot.quantity, len(future_prices), alpha, rng)
    revenue = float(np.sum(units_per_day * future_prices))   # сумма: единицы_дня * цена_дня
    profit = revenue - price_paid
    return {
        "units_per_day": units_per_day,
        "prices": future_prices,
        "revenue": revenue,
        "profit": profit,
        "average_sale_price": revenue / lot.quantity,
    }


def settle_lots(results, market, alpha, rng):
    """Распродаёт все купленные лоты, записывает прибыль агентам."""
    sold_lots = []
    for result in results:
        if result["winner"] is None:
            continue                      # лот не купили: продавать нечего

        lot = result["lot"]
        agent = result["winner"]
        future_prices = market.get_future(lot.product)
        sale = sell_lot(lot, result["price_paid"], future_prices, alpha, rng)

        agent.profit = agent.profit + sale["profit"]    # прибыль идёт агенту

        sold_lots.append({
            "lot_id": lot.lot_id,
            "product": lot.product,
            "quantity": lot.quantity,
            "agent": agent.name,
            "price_paid": result["price_paid"],
            "unit_price_paid": lot.unit_price(result["price_paid"]),
            "auction_step": result["winner_step"],
            "revenue": sale["revenue"],
            "profit": sale["profit"],
            "average_sale_price": sale["average_sale_price"],
            "units_per_day": sale["units_per_day"],
            "prices": sale["prices"],
        })
    return sold_lots


def calculate_estimation_errors(agents, lots, market):
    """Для каждой пары (агент, лот) сравнивает оценку fair с истинной fair."""
    rows = []
    for agent in agents:
        for lot in lots:
            estimate = agent.get_fair_estimate(lot)
            fair_true = market.get_true_fair(lot.product)   # истина: только здесь!
            signed_error = (estimate - fair_true) / fair_true
            rows.append({
                "agent": agent.name,
                "lot_id": lot.lot_id,
                "product": lot.product,
                "fair_estimate": estimate,
                "fair_true": fair_true,
                "signed_error": signed_error,         # < 0: занижает, > 0: завышает
                "abs_error": abs(signed_error),
            })
    return rows


def build_agent_summary(agents, sold_lots, error_rows):
    rows = []
    for agent in agents:
        my_sales = [s for s in sold_lots if s["agent"] == agent.name]
        my_errors = [e for e in error_rows if e["agent"] == agent.name]

        spent = sum(s["price_paid"] for s in my_sales)
        revenue = sum(s["revenue"] for s in my_sales)

        rows.append({
            "agent": agent.name,
            "strategy": agent.strategy.name,
            "estimator": agent.estimator.name,
            "initial_budget": agent.initial_budget,
            "lots_bought": len(my_sales),
            "spent": spent,
            "revenue": revenue,
            "final_budget": agent.initial_budget - spent + revenue,
            "profit": agent.profit,
            "mean_abs_error": float(np.mean([e["abs_error"] for e in my_errors])),
            "mean_signed_error": float(np.mean([e["signed_error"] for e in my_errors])),
        })
    return pd.DataFrame(rows)

# 7. Оформление графиков
#
# На третьем графике точки показывают шаги, на которых агенты согласились купить лот. Чтобы подписи не перекрывали друг друга, названия и номера шагов указаны в легенде, а возле самих точек текстовых подписей нет.


# Каждому агенту свой цвет И своя форма маркера (форма нужна, чтобы агентов
# можно было различить даже без цвета)
AGENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AGENT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
NEUTRAL = "#898781"
LABEL_BOX = dict(facecolor="#fcfcfb", edgecolor="none", alpha=0.85, pad=1.5)


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10,
    })


def get_agent_style(agents):
    """Возвращает {имя агента: (цвет, маркер)}. Цвет привязан к агенту, а не к его месту в рейтинге."""
    style = {}
    for number, agent in enumerate(agents):
        if number < len(AGENT_COLORS):
            style[agent.name] = (AGENT_COLORS[number], AGENT_MARKERS[number])
        else:
            style[agent.name] = (NEUTRAL, "o")    # агентов больше 8: остальные серые
    return style


def finish(fig, save_path, use_tight=True):
    """Подготавливает и сохраняет график; отображение запускается один раз в конце."""
    if use_tight:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_agent_profits(agents, save_path=None):
    setup_style()
    style = get_agent_style(agents)
    fig, ax = plt.subplots(figsize=(9, 5))

    names = [agent.name + "\n" + agent.strategy.name.replace("(", "\n(") for agent in agents]
    profits = [agent.profit for agent in agents]
    colors = [style[agent.name][0] for agent in agents]

    bars = ax.bar(names, profits, color=colors, width=0.55)
    ax.axhline(0, color=NEUTRAL, linewidth=1)
    ax.grid(axis="x", visible=False)

    for bar, profit in zip(bars, profits):          # число над (или под) каждым столбцом
        offset = 4 if profit >= 0 else -4
        align = "bottom" if profit >= 0 else "top"
        ax.annotate(f"{profit:+.2f}", (bar.get_x() + bar.get_width() / 2, profit),
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", va=align, color=INK)

    ax.set_ylabel("прибыль")
    ax.set_title("Прибыль агентов по итогам симуляции", loc="left")
    ax.margins(y=0.15)
    finish(fig, save_path)


def plot_sale_prices(sold_lots, agents, save_path=None):
    setup_style()
    style = get_agent_style(agents)
    count = len(sold_lots)
    if count == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Ни один лот не был куплен", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        finish(fig, save_path)
        return
    columns = 2 if count > 1 else 1
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 3.6 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, sold in zip(flat_axes, sold_lots):
        color, marker = style[sold["agent"]]
        days = np.arange(1, len(sold["prices"]) + 1)
        sizes = 12 + sold["units_per_day"] * 5       # чем больше продано за день, тем крупнее точка

        ax.vlines(days, sold["unit_price_paid"], sold["prices"], color=GRID, linewidth=0.8)
        ax.scatter(days, sold["prices"], s=sizes, color=INK_SECONDARY, alpha=0.75, zorder=3)
        ax.axhline(sold["unit_price_paid"], color=color, linestyle="--", linewidth=2)   # цена покупки
        ax.axhline(sold["average_sale_price"], color=INK, linewidth=1.2)                # средняя цена продажи

        ax.annotate(f"купил {sold['agent']}: {sold['unit_price_paid']:.2f}",
                    (days[-1], sold["unit_price_paid"]), xytext=(0, 6), textcoords="offset points",
                    ha="right", color=INK, fontsize=9, bbox=LABEL_BOX)
        ax.annotate(f"средняя цена продажи: {sold['average_sale_price']:.2f}",
                    (days[0], sold["average_sale_price"]), xytext=(0, -13), textcoords="offset points",
                    ha="left", color=INK, fontsize=9, bbox=LABEL_BOX)
        ax.set_title(f"Лот {sold['lot_id']} ({sold['product']}, {sold['quantity']} шт.), "
                     f"прибыль {sold['profit']:+.2f}", loc="left", fontsize=10)
        ax.set_xlabel("день распродажи")
        ax.set_ylabel("цена за единицу")

    for ax in flat_axes[count:]:                      # лишние пустые клетки скрываем
        ax.set_visible(False)
    fig.suptitle("Цены продажи по лотам (размер точки = сколько единиц продано в этот день)",
                 x=0.01, ha="left")
    finish(fig, save_path)


def plot_lot_dynamics(result, market, agents, save_path=None):
    """История цены, шаги аукциона и цена в течение распродажи."""
    setup_style()
    style = get_agent_style(agents)
    lot = result["lot"]
    history = lot.history
    future = market.get_future(lot.product)
    fair_true = market.get_true_fair(lot.product)
    levels = lot.get_price_levels()
    auction_unit_prices = [lot.unit_price(price) for price in levels]
    steps = np.arange(len(levels))

    fig, (ax_history, ax_auction, ax_sales) = plt.subplots(
        1, 3, sharey=True, figsize=(14, 5),
        gridspec_kw={
            "width_ratios": [180, 70, 30],
            "wspace": 0.06,
            "left": 0.06, "right": 0.99, "top": 0.82, "bottom": 0.14,
        },
    )

    # 180 дней истории
    ax_history.plot(
        np.arange(1, len(history) + 1), history,
        color=INK_SECONDARY, linewidth=1.5,
    )
    ax_history.set_xlabel("день истории")
    ax_history.set_ylabel("цена за единицу")
    ax_history.set_title("История (180 дней)", loc="left", fontsize=10)

    # Шаги снижающегося аукциона
    ax_auction.plot(steps, auction_unit_prices, color=INK, linewidth=1.5)
    legend_handles = []
    for number, agent in enumerate(agents):
        color, marker = style[agent.name]
        ready_price = result["ready_prices"][agent.name]
        is_winner = result["winner"] is not None and result["winner"].name == agent.name

        if ready_price is None:
            label = f"{agent.name} — не согласился"
        else:
            step_number = min(
                range(len(levels)),
                key=lambda i: abs(levels[i] - ready_price),
            )
            label = f"{agent.name} — шаг {step_number}"
            if is_winner:
                label += " (купил)"

            ax_auction.scatter(
                [step_number], [lot.unit_price(ready_price)],
                s=100, marker=marker, color=color,
                edgecolor=INK if is_winner else "#fcfcfb",
                linewidth=2 if is_winner else 1.2,
                zorder=4 + number, clip_on=False,
            )

        # В легенде сразу указано, на каком шаге агент согласился.
        legend_handles.append(
            plt.Line2D(
                [0], [0], marker=marker, linestyle="None",
                markerfacecolor=color, markeredgecolor=INK if is_winner else color,
                markersize=7, label=label,
            )
        )

    ax_auction.set_xlim(-1, max(steps[-1], 1) + 1)
    ax_auction.set_xlabel("шаг аукциона")
    ax_auction.set_title("Аукцион (цена падает)", loc="left", fontsize=10)

    # 30 дней распродажи
    ax_sales.plot(
        np.arange(1, len(future) + 1), future,
        color=INK_SECONDARY, linewidth=1.5,
    )
    ax_sales.set_xlabel("день распродажи")
    ax_sales.set_title("30 дней после", loc="left", fontsize=10)

    for ax in (ax_history, ax_auction, ax_sales):
        ax.axhline(fair_true, color=NEUTRAL, linestyle=":", linewidth=1.2)

    ax_auction.legend(
        handles=legend_handles, title="Агенты и шаг согласия",
        frameon=False, loc="best", fontsize=8, title_fontsize=9,
    )

    for ax in (ax_auction, ax_sales):
        ax.tick_params(left=False)
        ax.spines["left"].set_visible(False)

    fig.suptitle(
        f"Лот {lot.lot_id} ({lot.product}, {lot.quantity} шт.): цена за единицу",
        x=0.01, ha="left",
    )
    finish(fig, save_path, use_tight=False)


def plot_fair_price_by_lot(lots, market, save_path=None):
    """Показывает историю цены каждого лота и истинную fair-цену товара."""
    setup_style()
    count = len(lots)
    columns = 2 if count > 1 else 1
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(6.2 * columns, 3.8 * rows), squeeze=False
    )
    flat_axes = axes.flatten()

    for ax, lot in zip(flat_axes, lots):
        fair_true = market.get_true_fair(lot.product)
        days = np.arange(1, len(lot.history) + 1)
        ax.plot(days, lot.history, color=INK_SECONDARY, linewidth=1.5,
                label="Рыночная цена")
        ax.axhline(fair_true, color=NEUTRAL, linestyle=":", linewidth=1.8,
                   label=f"Справедливая цена = {fair_true:.2f}")
        ax.scatter([days[-1]], [lot.current_market_price], color=INK,
                   s=35, zorder=3, label="Цена на момент аукциона")
        ax.set_title(f"Лот {lot.lot_id}: {lot.product}, {lot.quantity} шт.",
                     loc="left", fontsize=10)
        ax.set_xlabel("день истории")
        ax.set_ylabel("цена за единицу")
        ax.legend(frameon=False, fontsize=8)

    for ax in flat_axes[count:]:
        ax.set_visible(False)

    fig.suptitle("История рыночной цены и справедливая цена каждого лота",
                 x=0.01, ha="left")
    finish(fig, save_path)

# 8. Главный запуск и сохранение данных
#
# Запуск создаёт рынок, лоты и агентов заново, проводит аукцион, рассчитывает прибыль и ошибки оценки, строит три графика и сохраняет таблицы CSV плюс общий JSON-файл.
#
# Файлы сохраняются в папку auction_output рядом с ноутбуком.


import json
from pathlib import Path


SEED = 42
DIRICHLET_ALPHA = 1
MIN_PRICE_SHARE_LOT = 0.5
STEP_SHARE = 0.01

LOTS = [
    ("роза", 100),
    ("пион", 50),
    ("тюльпан", 200),
    ("роза", 80),
]


def run_simulation(seed=SEED, output_dir=None, make_plots=True):
    """Полный запуск симуляции и сохранение результатов в CSV/JSON."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "auction_output"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Каждый тип случайности получает свой генератор.
    market = Market(PRODUCTS, seed=seed)
    lots = create_lots(market, LOTS, MIN_PRICE_SHARE_LOT, STEP_SHARE)
    strategy_rng = np.random.default_rng(seed + 1000)
    auction_rng = np.random.default_rng(seed + 2000)
    sales_rng = np.random.default_rng(seed + 3000)

    agents = [
        Agent("A1", 10000, AggressiveStrategy(), MeanEstimator(180)),
        Agent("A2", 10000, CautiousStrategy(0.95), MedianEstimator(30)),
        Agent("A3", 10000, RandomStrategy(0.2, strategy_rng), EmaEstimator(0.05)),
        Agent("A4", 10000, MovingAverageStrategy(30), MeanEstimator(30)),
        Agent("A5", 10000, WaitForDiscountStrategy(0.15), MeanEstimator(7)),
    ]

    auction = Auction(lots, agents, auction_rng)
    auction_results = auction.run_all()
    sold_lots = settle_lots(auction_results, market, DIRICHLET_ALPHA, sales_rng)
    error_rows = calculate_estimation_errors(agents, lots, market)
    summary = build_agent_summary(agents, sold_lots, error_rows)

    # Таблицы CSV
    summary.to_csv(output_path / "agents_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows).to_csv(
        output_path / "estimation_errors.csv", index=False, encoding="utf-8-sig"
    )

    sold_lots_csv = [
        {key: value for key, value in sold.items() if key not in ("units_per_day", "prices")}
        for sold in sold_lots
    ]
    pd.DataFrame(sold_lots_csv).to_csv(
        output_path / "sold_lots.csv", index=False, encoding="utf-8-sig"
    )

    auction_rows = []
    for result in auction_results:
        lot = result["lot"]
        winner = result["winner"]
        auction_rows.append({
            "lot_id": lot.lot_id,
            "product": lot.product,
            "quantity": lot.quantity,
            "start_price": lot.start_price,
            "min_price": lot.min_price,
            "step": lot.step,
            "winner": winner.name if winner else None,
            "price_paid": result["price_paid"],
            "winner_step": result["winner_step"],
            "ready_prices": result["ready_prices"],
        })
    pd.DataFrame([
        {**row, "ready_prices": json.dumps(row["ready_prices"], ensure_ascii=False)}
        for row in auction_rows
    ]).to_csv(output_path / "auction_results.csv", index=False, encoding="utf-8-sig")

    # JSON: все основные результаты, включая дневные продажи и цены.
    json_sold_lots = []
    for sold in sold_lots:
        json_sold_lots.append({
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in sold.items()
        })

    report = {
        "seed": seed,
        "settings": {
            "history_days": HISTORY_DAYS,
            "sell_days": SELL_DAYS,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "min_price_share_lot": MIN_PRICE_SHARE_LOT,
            "step_share": STEP_SHARE,
        },
        "auction_results": auction_rows,
        "agents_summary": summary.to_dict(orient="records"),
        "sold_lots": json_sold_lots,
        "estimation_errors": error_rows,
        "totals": {
            "auction_revenue": float(sum(s["price_paid"] for s in sold_lots)),
            "resale_revenue": float(sum(s["revenue"] for s in sold_lots)),
            "total_profit": float(sum(agent.profit for agent in agents)),
        },
    }
    with open(output_path / "results.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=float)

    # Понятный вывод результатов в консоль
    print("РЕЗУЛЬТАТЫ АУКЦИОНА")
    for result in auction_results:
        lot = result["lot"]
        if result["winner"] is None:
            print(f"Лот {lot.lot_id} ({lot.product}): не продан")
        else:
            print(
                f"Лот {lot.lot_id} ({lot.product}, {lot.quantity} шт.): "
                f"{result['winner'].name} купил за {result['price_paid']:.2f} "
                f"(шаг {result['winner_step']})"
            )

    print("\nИТОГИ ПО АГЕНТАМ")
    print(summary.round(3).to_string(index=False))
    print(f"Выручка аукциона: {report['totals']['auction_revenue']:.2f}")
    print(f"Выручка от перепродажи: {report['totals']['resale_revenue']:.2f}")
    print(f"Суммарная прибыль агентов: {report['totals']['total_profit']:.2f}")
    print(f"\nРезультаты сохранены в: {output_path.resolve()}")

    if make_plots:
        plot_agent_profits(agents, output_path / "agent_profits.png")
        plot_sale_prices(sold_lots, agents, output_path / "sale_prices.png")
        # Выбираем второй лот, если он есть, иначе первый.
        chosen_result = auction_results[1] if len(auction_results) > 1 else auction_results[0]
        plot_lot_dynamics(
            chosen_result, market, agents, output_path / "lot_dynamics.png"
        )
        plot_fair_price_by_lot(lots, market, output_path / "fair_prices_by_lot.png")
        # Показываем все окна одновременно, а не по очереди.
        plt.show()

    return {
        "market": market,
        "lots": lots,
        "agents": agents,
        "auction_results": auction_results,
        "sold_lots": sold_lots,
        "error_rows": error_rows,
        "summary": summary,
        "report": report,
    }



# Что сохраняется
#
# - agents_summary.csv — прибыль, выручка и средняя ошибка оценки каждого агента.
# - estimation_errors.csv — ошибки оценивания для каждой пары «агент — лот».
# - sold_lots.csv — результаты распродажи купленных лотов.
# - auction_results.csv — победитель, цена покупки и шаг согласия по каждому лоту.
# - results.json — общий отчёт, включая историю продаж по дням и итоговые показатели.
# - agent_profits.png, sale_prices.png, lot_dynamics.png, fair_prices_by_lot.png — сохранённые графики.


if __name__ == "__main__":
    run_simulation()