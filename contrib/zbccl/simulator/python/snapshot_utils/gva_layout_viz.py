import matplotlib.pyplot as plt
from .gva_layout_analyzer import MemoryAllocatorSimulator


def visualize_memory(simulator: MemoryAllocatorSimulator, display_segments=False, display_blocks=True):
    # 绘图准备
    fig, ax = plt.subplots(figsize=(12, 2))

    # 绘制总背景（空闲内存）
    ax.add_patch(plt.Rectangle((simulator.gva_base_addr, 0), simulator.gva_size, 1, color='#f0f0f0', label='Free'))

    # 绘制已分配segments
    if display_segments:
        for start_addr, end_addr in simulator.get_all_segments():
            ax.broken_barh([(start_addr, end_addr-start_addr)], (0.05, 0.9), facecolors='#D3D3D3', edgecolor='black', alpha=0.8)

    # 绘制已分配blocks
    if display_blocks:
        for used_block in simulator.get_used_blocks():
            ax.broken_barh([(used_block.addr, used_block.size)], (0.1, 0.8), facecolors='#4CAF50', edgecolor='black', alpha=0.8)

    # 设置坐标轴
    ax.set_xlim(simulator.gva_base_addr, simulator.gva_base_addr + simulator.gva_size)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    # 格式化坐标轴为 GB 或 Hex
    def format_func(value, tick_number):
        gb_val = (value - simulator.gva_base_addr) / (1024 ** 3)
        return f"{gb_val:.1f}G"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    plt.title(f"Memory Layout Visualization (Total: {simulator.gva_size / (1024 ** 3)}GB)")
    plt.xlabel("Address")

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#f0f0f0', lw=4, label='Unused')]
    if display_segments:
        legend_elements.append(Line2D([0], [0], color='#f0f0f0', lw=4, label='Allocated Segment'))
    if display_blocks:
        legend_elements.append(Line2D([0], [0], color='#f0f0f0', lw=4, label='Allocated Block'))
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

