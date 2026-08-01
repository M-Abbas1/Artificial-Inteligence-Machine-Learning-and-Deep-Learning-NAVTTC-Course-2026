class INT:
    def __init__(self, val):
        self.value = val

    def __str__(self):
        return f"{self.value} year"

    def __gt__(self, other):
        return self.value > other.value
    
    def __add__(self, other):
        return self.value + other.value

