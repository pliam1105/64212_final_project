from state_machine import StateMachine
from pydrake.all import StartMeshcat
import time
import sys

def main():
    sm = StateMachine()
    print("initial state:", sm.state)
    sm.step()  # preRANSAC
    print("after preRANSAC:", sm.state)
    sm.step()  # open_drawer
    print("after open_drawer:", sm.state)
    sm.step()  # SAM
    print("final state:", sm.state)
    
    # Keep Python process alive while MeshCat is being viewed
    if sm.meshcat:
        print("\n" + "="*80)
        print("[main] Simulation complete!")
        print("[main] MeshCat is ready for viewing.")
        print("[main] Keeping Python process alive for MeshCat interaction...")
        print("[main] Press Ctrl+C to exit when done viewing")
        print("="*80 + "\n")
        
        try:
            # Keep the process running indefinitely
            # This allows users to view and interact with MeshCat
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[main] Received interrupt signal. Exiting gracefully...")
            sys.exit(0)
    else:
        print("[main] No MeshCat. Exiting.")

if __name__ == '__main__':
    main()