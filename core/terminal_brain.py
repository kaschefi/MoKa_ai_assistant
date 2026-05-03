from datetime import datetime
from core.semantic_layer import check_layer_1
from core.router import run_cozmo_agent


def terminal_chat():
    print("=======================================")
    print("              COZMO BRAIN ")
    print("=======================================\n")
    print("Type 'quit' to exit.\n")

    while True:
        command = input("You: ")
        if command.lower() in ['quit', 'exit']:
            print("Shutting down brain...")
            break

        print("\nProcessing...")

        layer_1_route = check_layer_1(command)

        if layer_1_route:
            print(f"[LAYER 1 TRIGGERED]: Route -> '{layer_1_route}'")

            if layer_1_route == "get_date":
                today = datetime.now().strftime("%A, %B %d, %Y")
                print(f" Cozmo: Today is {today}.")
            elif layer_1_route == "dock_with_charger":
                print(" Cozmo: Heading back to base!")
                print(" [Hardware Mock]: Disabling AI, triggering wheel motors...")
            elif layer_1_route == "tell_joke":
                print(" Cozmo: Why do robots never get scared? Because they have nerves of steel!")

            print("---------------------------------------\n")
            continue

        print(" [LAYER 1 FAILED]: No reflex matched. Routing to LangGraph (Layer 2)...")
        final_answer = run_cozmo_agent(command)

        print(f"Cozmo: {final_answer}")
        print("---------------------------------------\n")


if __name__ == "__main__":
    terminal_chat()