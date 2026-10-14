import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_clean_context_diagram():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Настройки стилей
    box_conf = dict(linewidth=1.5, edgecolor='#4B0082', facecolor='#F0E6FF')
    system_conf = dict(linewidth=2.5, edgecolor='#4B0082', facecolor='white')
    font_main = dict(fontsize=10, fontweight='bold', ha='center', va='center')
    font_labels = dict(fontsize=8, color='#00008B', ha='center', va='center')

    # 1. Отрисовка системы (Центр)
    system_center = (5, 5)
    ellipse = patches.Ellipse(system_center, 2.8, 1.4, **system_conf)
    ax.add_patch(ellipse)
    ax.text(5, 5, "СИСТЕМА\nДОСТАВКИ", **font_main)

    # 2. Отрисовка сущностей (Координаты выверены для ровных стрелок)
    entities = {
        'operator': (5, 8.5, "ОПЕРАТОР"),
        'admin': (5, 1.5, "АДМИНИСТРАТОР"),
        'dispatcher': (1.2, 5, "ДИСПЕТЧЕР"),
        'courier': (8.8, 5, "КУРЬЕР"),
        'accounting': (8.8, 8.5, "БУХГАЛТЕРИЯ\n(Внешняя ИС)"),
        'shops': (1.2, 8.5, "МАГАЗИНЫ / КАФЕ")
    }

    rects = {}
    for key, (x, y, name) in entities.items():
        rect = patches.Rectangle((x - 0.9, y - 0.4), 1.8, 0.8, **box_conf)
        ax.add_patch(rect)
        ax.text(x, y, name, **font_main)
        rects[key] = (x, y)

    # 3. Отрисовка СТРЕЛОК (Используем смещения для параллельности)

    # Оператор -> Система (Вертикально вниз)
    ax.annotate("Данные заказов\n(единый формат)", xy=(5, 5.75), xytext=(5, 8.1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), **font_labels)

    # Администратор -> Система (Вертикально вверх)
    ax.annotate("Права доступа,\nрегистрация курьеров", xy=(5, 4.25), xytext=(5, 1.9),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), **font_labels)

    # Диспетчер <-> Система (Горизонтально)
    ax.annotate("Переназначение", xy=(3.55, 5.2), xytext=(2.1, 5.2),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), **font_labels)
    ax.annotate("Мониторинг", xy=(2.1, 4.8), xytext=(3.55, 4.8),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), **font_labels)

    # Курьер <-> Система (Горизонтально)
    ax.annotate("Статусы, Бронь", xy=(6.45, 5.2), xytext=(7.9, 5.2),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), **font_labels)
    ax.annotate("Список заказов,\nОплата", xy=(7.9, 4.8), xytext=(6.45, 4.8),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), **font_labels)

    # Система <-> Бухгалтерия (Диагональ)
    ax.annotate("Данные для расчета", xy=(8.2, 8.1), xytext=(6.0, 5.6),
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                ha='left', va='bottom', fontsize=8, color='#00008B')
    ax.annotate("Суммы к выплате", xy=(6.2, 5.4), xytext=(8.4, 7.9),
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                ha='right', va='top', fontsize=8, color='#00008B')

    # Взаимодействие вне системы: Магазины -> Оператор
    ax.annotate("Заказы (тел/чат)", xy=(4.05, 8.5), xytext=(2.1, 8.5),
                arrowprops=dict(arrowstyle="->", color="gray", linestyle='dashed'), **font_labels)

    plt.title("Контекстная диаграмма: Система доставки (TO BE)", fontsize=14, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex01_DELIV_context_fixed.png', dpi=300)
    plt.show()


draw_clean_context_diagram()