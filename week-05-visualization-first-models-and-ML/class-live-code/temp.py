import matplotlib.pyplot as plt

day = ['mon', 'tue', 'wen', 'thu', 'fri', 'sat', 'sun']
temp = [30.6, 40.0, 29.8, 25.5, 30.5, 33.6, 42.5]


plt.plot(day, temp)
plt.title("Weekly Temprature")
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.show()