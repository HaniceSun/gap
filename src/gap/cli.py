import argparse
from importlib import resources
from .ancestry import Ancestry
from .models import GapLearn
import pandas as pd

def get_parser():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest='command', required=True)

    p1 = subparsers.add_parser("get-reference-data", help="get reference data with ancestry labeled")
    p1.add_argument('--source', type=str, default='1000genomes', help='data source to download from')
    p1.add_argument('--output_dir', type=str, default='data', help='output directory to save the downloaded data')

    p2 = subparsers.add_parser('merge-dataset-with-reference', help='merge your dataset with the reference data')
    p2.add_argument('--dataset', type=str, required=True, help='your input vcf file, must be on GRCh38 build and the REF allele must be consistent with the reference genome')
    p2.add_argument('--reference', type=str, default='data/1000genomes_unrelated.vcf.gz', help='the vcf file of the reference data downloaded previously')
    p2.add_argument('--output', type=str, default='data/merged', help='the merged file, prefix only')
    p2.add_argument('--threads', type=int, default=4, help='number of threads to use')

    p3 = subparsers.add_parser('feature-engineering', help='feature engineering for ancestry prediction')
    p3.add_argument('--input', type=str, default='data/merged', help='the merged file from the previous step, prefix only')
    p3.add_argument('--output', type=str, default='data/features.txt', help='the output feature matrix file')
    p3.add_argument('--pruning', type=str, default='True', help='remove variants that are highly correlated')
    p3.add_argument('--pruning_params', type=str, default='50,5,0.2', help='window size, step size, r2 threshold for pruning')
    p3.add_argument('--pca', type=str, default='True', help='dimensionality reduction using PCA')
    p3.add_argument('--pca_params', type=str, default='50', help='first n principal components to keep')
    p3.add_argument('--threads', type=int, default=4, help='number of threads to use')

    p4 = subparsers.add_parser('add-labels', help='add ancestry labels to the feature matrix')
    p4.add_argument('--feature_file', type=str, default='data/features.txt', help='the feature matrix file')
    p4.add_argument('--label_file', type=str, default='data/1000genomes_unrelated_sampleInfo.txt', help='the file containing ancestry labels for the reference samples')

    p5 = subparsers.add_parser('split-train-test', help='split the dataset into train and test sets')
    p5.add_argument('--input', type=str, default='data/features_labeled_Superpopulation.txt', help='the input file with features and labels')
    p5.add_argument('--test_size', type=float, default=0.2, help='proportion of the dataset to include in the test split')

    p6 = subparsers.add_parser('train-model', help='train models with different parameters')
    p6.add_argument('--config_file', type=str, default='config.yaml', help='the configuration file specifying models and parameters')
    p6.add_argument('--train_file', type=str, default='data/features_labeled_Superpopulation_train.txt', help='the training set file with features and labels')
    p6.add_argument('--test_file', type=str, default='data/features_labeled_Superpopulation_test.txt', help='the test set file with features and labels')
    p6.add_argument('--metrics_file', type=str, default='data/metrics_Superpopulatoiin.txt', help='the output file to save model performance metrics')
    p6.add_argument('--n_features', type=str, default='10,20,30', help='the number of features to use for training')

    p7 = subparsers.add_parser('eval-model', help='evaluate models and select the best one')
    p7.add_argument('--config_file', type=str, default='config.yaml', help='the configuration file specifying models and parameters')
    p7.add_argument('--train_file', type=str, default='data/features_labeled_Superpopulation_train.txt', help='the training set file with features and labels')
    p7.add_argument('--test_file', type=str, default='data/features_labeled_Superpopulation_test.txt', help='the test set file with features and labels')
    p7.add_argument('--metrics_file', type=str, default='data/metrics_Superpopulatoiin.txt', help='the model performance metrics')
    p7.add_argument('--model_file', type=str, default='model_Superpopulation.pkl', help='the file containing the trained model')

    p8 = subparsers.add_parser('predict', help='predict ancestry using the best model in the validation step')
    p8.add_argument('--input', type=str, default='ToPred.txt', help='the input feature matrix file for prediction')
    p8.add_argument('--output', type=str, default='predicted_Superpopulation.txt', help='the output file to save predictions')
    p8.add_argument('--label_file', type=str, default='data/labels_Superpopulation.txt', help='the label file to decode labels')
    p8.add_argument('--metrics_file', type=str, default='data/metrics_Superpopulation.txt', help='the model performance metrics')
    p8.add_argument('--model_file', type=str, default='model_Superpopulation.pkl', help='the file containing the trained model')
    p8.add_argument('--model_name', type=str, default=None, help='the model name to use for prediction, using the best model if not specified')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    an = Ancestry()
    if args.command == 'get-reference-data':
        an.get_reference_data(source=args.source, out_dir=args.output_dir)
    elif args.command == 'merge-dataset-with-reference':
        an.merge_dataset_with_reference(dataset_vcf=args.dataset, reference_vcf=args.reference, out_file=args.output, threads=args.threads)
    elif args.command == 'feature-engineering':
        pruning = (args.pruning.lower() == 'true')
        pruning_params = [float(x) for x in args.pruning_params.split(',')]
        pca = (args.pca.lower() == 'true')
        pca_params = [int(x) for x in args.pca_params.split(',')]
        an.feature_engineering(in_file=args.input, out_file=args.output, pruning=pruning, pruning_params=pruning_params,
                               pca=pca, pca_params=pca_params, threads=args.threads)
    elif args.command == 'add-labels':
        an.add_labels(in_file=args.feature_file, label_file=args.label_file)
    elif args.command == 'split-train-test':
        an.split_train_test(in_file=args.input, test_size=args.test_size)
    elif args.command == 'train-model':
        gl = GapLearn()
        gl.train_cross_val(config_file=args.config_file, train_file=args.train_file, test_file=args.test_file,
                           metrics_file=args.metrics_file, n_features=args.n_features)
    elif args.command == 'eval-model':
        gl = GapLearn()
        gl.get_best_model_and_params(metrics_file=args.metrics_file)
        gl.final_fit_eval_on_full_train_then_eval_on_test(config_file=args.config_file, in_file=args.metrics_file.replace('.txt', '_sorted_best.txt'),
                                                          train_file=args.train_file, test_file=args.test_file, model_file=args.model_file)
    elif args.command == 'predict':
        gl = GapLearn()
        model_name = args.model_name
        if not model_name:
            try:
                metrics_file = args.metrics_file.replace('.txt', '_sorted_best.txt')
                model_name = pd.read_table(metrics_file, header=0, sep='\t')['model'].iloc[0]
            except Exception as e:
                print(f"Cannot get the best model from metrics file {e}")
        print('Using model:', model_name)
        gl.predict(in_file=args.input, out_file=args.output, label_file=args.label_file, model_name=model_name, model_file=args.model_file)

if __name__ == '__main__':
    main()
