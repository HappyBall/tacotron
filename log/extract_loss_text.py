import sys

def main():
    if len(sys.argv) != 3:
        print("Usage:", sys.argv[0], "<log file> <output file>")
        sys.exit(1)

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    f = open(in_file, "r")
    opf = open(out_file, "w")

    while True:
        line = f.readline()

        if not line:
            break

        ind = line.find("Epoch")
        if ind != -1:
            opf.write(line[ind:])

if __name__ == "__main__":
    main()
