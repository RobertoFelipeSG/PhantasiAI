from app.main import RAGApplication


def main():
    try:
        app = RAGApplication()
        app.initialize_database()
        app.run_interactive()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
