import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_net():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Function to draw a box with text
    def draw_box(x, y, width, height, color, label, subtext):
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2 + 0.3, label, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x + width/2, y + height/2 - 0.3, subtext, ha='center', va='center', fontsize=10)
        return (x + width, y + height/2) # Return connection point (right side)

    # 1. INPUT LAYER
    input_x, input_y = 0.5, 2.0
    out_in = draw_box(input_x, input_y, 2.0, 2.0, '#E1F5FE', "Input Sequence", "24 Time Steps\n60 Features\n(Load, Weather, Prev State)")

    # 2. HIDDEN LAYERS (STACKED GRU)
    gru_x = 4.0
    # Layer 1
    draw_box(gru_x, 0.5, 2.5, 1.0, '#FFF9C4', "GRU Layer 1", "256 Hidden Units")
    # Layer 2
    draw_box(gru_x, 2.0, 2.5, 1.0, '#FFF9C4', "GRU Layer 2", "256 Hidden Units")
    # Layer 3
    out_gru = draw_box(gru_x, 3.5, 2.5, 1.0, '#FFF9C4', "GRU Layer 3", "256 Hidden Units")

    # Arrows for GRU Stack
    ax.annotate("", xy=(gru_x+1.25, 2.0), xytext=(gru_x+1.25, 1.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(gru_x+1.25, 3.5), xytext=(gru_x+1.25, 3.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # Arrow from Input to GRU
    ax.annotate("", xy=(gru_x, 2.5), xytext=(out_in[0], 2.5), arrowprops=dict(arrowstyle="->", lw=1.5, color='blue'))
    ax.text(3.25, 2.7, "Forward\nPass", ha='center', fontsize=9)

    # 3. FULLY CONNECTED LAYER
    fc_x = 8.0
    out_fc = draw_box(fc_x, 2.5, 1.5, 1.0, '#E0F2F1', "Linear Layer", "Fully Connected")

    # Arrow from GRU to FC (Only last time step)
    ax.annotate("", xy=(fc_x, 3.0), xytext=(out_gru[0], 4.0), arrowprops=dict(arrowstyle="->", lw=1.5, color='red'))
    ax.text(7.25, 3.8, "Last Hidden\nState ($h_T$)", ha='center', fontsize=9)

    # 4. OUTPUT LAYER
    out_x = 10.5
    draw_box(out_x, 2.0, 1.5, 2.0, '#FFEBEE', "Output ($Y_t$)", "35 Variables\n(P, Q, V, E)")

    # Arrow from FC to Output
    ax.annotate("", xy=(out_x, 3.0), xytext=(out_fc[0], 3.0), arrowprops=dict(arrowstyle="->", lw=1.5))

    plt.title("Proposed GRU Network Architecture", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("GRU_Architecture.png", dpi=300)
    print("Diagram saved as GRU_Architecture.png")
    plt.show()

if __name__ == "__main__":
    draw_neural_net()