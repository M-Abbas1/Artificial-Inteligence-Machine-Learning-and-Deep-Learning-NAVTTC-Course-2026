## A simple Student class Example

class Student:
    def __init__(self, name, age, gpa):
        self.name = name
        self._age = age      # protected
        self.__gpa = gpa     ## private
        self.courses = []

    def show_detail(self):
        print("----student data-----")
        print(f"Name : {self.name}")
        print(f"Age : {self._age}")
        print(f"GPA : {self.__gpa}")

    @property
    def name(self):
        return self.name

    @name.setter
    def name(self, n):
        self.__name = n


print("Creating First Object of Student Class")
s1 = Student("Ahmad", 32, 3.4)
s1.show_detail()


print()


print("Creating second object of student class")
Student("Salman", 21, 3.4)
s1.show_detail()



