from src import parse_arguments, generate_dataset, train, test

args = parse_arguments()
mode = args.mode

if mode == "data":
    generate_dataset(args)  # train set
    generate_dataset(args, test_dataset=True)  # test set
elif mode == "train":
    train(args)
elif mode == "test":
    test(args)
else:
    raise(ValueError("Mode %s is not recognized" % mode))

