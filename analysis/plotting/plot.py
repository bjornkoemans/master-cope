import matplotlib.pyplot as plt


def main():
    plt.figure()
    # Per agent plot broken bar
    plt.broken_barh(xranges=[(2, 4), (7, 10)], yrange=(0, 1), facecolors="blue")
    plt.broken_barh(xranges=[(1, 2), (5, 7)], yrange=(1, 1), facecolors="red")
    plt.broken_barh(xranges=[(0, 1), (3, 5)], yrange=(2, 1), facecolors="green")
    # Y ticks show agent ids
    plt.yticks([0.5, 1.5, 2.5], ["agent_0", "agent_1", "agent_2"])
    # X ticks show time steps
    plt.xlabel("Time step")
    plt.ylabel("Agent ID")
    plt.title("Agent availability over time")
    plt.plot()
    plt.show()


if __name__ == "__main__":
    main()
