from matplotlib import pyplot as plt


def mapVal(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

db = []
p = []
for i in range(101):
    volDB = mapVal(i, 0, 100, -65.25, 0.0)
    db.append(volDB)
    p.append(i)


plt.plot(db, p, color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12) 
plt.show()