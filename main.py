from src import (parse_arguments,
                 generate_dataset,
                 train,
                 test, test_exotic_shape,
                 viz)

args = parse_arguments()
mode = args.mode

if mode == "generate_data":
    generate_dataset(args)  # train set
    generate_dataset(args, test_dataset=True)  # test set
elif mode == "train":
    train(args)
elif mode == "test_exotic":
    test_exotic_shape(args)
elif mode == "test":
    test(args)
elif mode == "visualize":
    viz(args)
else:
    raise(ValueError("Mode %s is not recognized" % mode))

