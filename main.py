import openollama


def main():
    openollama.app.debug = True
    openollama.app.run(threaded=True)


if __name__ == "__main__":
    main()
