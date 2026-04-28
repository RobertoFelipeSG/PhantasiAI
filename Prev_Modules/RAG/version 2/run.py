from main import RAGApplication
import traceback


def main():
    try:
        app = RAGApplication()
        app.initialize_database()
        app.run_interactive()
    except Exception as e:
        # print(f"Error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
