from transitions import Machine


class A:

    def __init__(self, boolean, boolean_2):
        self.boolean = boolean
        self.boolean_2 = boolean_2

    def check_b(self):
        return self.boolean

    def check_b2(self):
        return self.boolean_2


class C(A):
    def __init__(self):
        super().__init__(True, True)
        self.fsm = Machine(model=self, states=["1", "2", "3", "4", "5"], initial="1")
        self.fsm.add_transition(
            "run", "1", "2", unless=[self.check_b, self.check_b2], after=[self.check_b]
        )
        self.fsm.add_transition(
            "run",
            "1",
            "3",
            conditions=[self.check_b2],
            unless=[self.check_b],
        )
        self.fsm.add_transition(
            "run",
            "1",
            "4",
            conditions=[self.check_b],
            unless=[self.check_b2],
        )
        self.fsm.add_transition(
            "run",
            "1",
            "5",
            conditions=[self.check_b, self.check_b2],
        )


class B(Machine):

    def __init__(self, boolean, boolean_2):
        Machine.__init__(self, states=["1", "2", "3", "4", "5"], initial="1")
        self.boolean = boolean
        self.boolean_2 = boolean_2

    def check_b(self):
        print("check")
        return self.boolean

    def check_b2(self):
        print("check_2")
        return self.boolean_2


# b = B(True, False)

# # print(b.state)

# # # b.trigger("run")
# # b.run()

# # print(b.state)

# match (b.check_b(), b.check_b2()):
#     case (True, True):
#         print("tt")
#     case (True, False):
#         print("tf")

# print(b.get_transitions(source="1", dest="2"))
# b.get_transitions(source="1", dest="2")[0].after.append(b.check_b)
# print(b.get_transitions(source="1", dest="2"))
# b.to_2()

c = C()
print(c.state)
# print(f"fsm {c.fsm.state}")

# print(c.get_transitions(source="1", dest="2"))
print(f"fsm {[t for t in c.fsm.get_transitions() if t.source != t.dest]}")

c.fsm.trigger("run")
