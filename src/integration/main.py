from state_machine import StateMachine
from pydrake.all import StartMeshcat

def main():
    sm = StateMachine()
    print("initial state:", sm.state)
    sm.step()  # preRANSAC
    print("after preRANSAC:", sm.state)
    sm.step()  # open_drawer
    print("after open_drawer:", sm.state)
    # sm.step()  # home
    # print("final state:", sm.state)

if __name__ == '__main__':
    main()