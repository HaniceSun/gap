import os
import pandas as pd
import subprocess
from .models import GapLearn
from sklearn.model_selection import train_test_split

class Ancestry:
    def __init__(self):
        self.url_1000genomes_vcf = 'http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/' \
                + 'working/20201028_3202_phased/CCDG_14151_B01_GRM_WGS_2020-08-05_{}.filtered.shapeit2-duohmm-phased.vcf.gz'

        self.url_1000genomes_sample_populatioin = 'http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/' \
                + '1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt'

        self.url_1000genomes_sample_unrelated = 'http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/' \
                + '1000G_2504_high_coverage/1000G_2504_high_coverage.sequence.index'

        self.chs = [f'chr{i}' for i in range(1, 23)]

    def get_reference_data(self, out_dir='data', source='1000genomes', url=None, unrelated_samples_only=True):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if source == '1000genomes':
            if not url:
                self.url = self.url_1000genomes_vcf
                vcfs = []
                for ch in self.chs:
                    in_file = self.url.format(ch)
                    in_file_idx = in_file + '.tbi'
                    out_file = os.path.join(out_dir, f'{source}_{ch}.vcf.gz')
                    out_file_idx = os.path.join(out_dir, f'{source}_{ch}.vcf.gz.tbi')
                    try:
                        cmd = f'curl {in_file} -o {out_file}'
                        cmd_idx = f'curl {in_file_idx} -o {out_file_idx}'
                        subprocess.run(cmd, shell=True, check=True)
                        subprocess.run(cmd_idx, shell=True, check=True)
                        vcfs.append(out_file)
                    except Exception as e:
                        print(f"Error downloading {in_file}: {e}")

            out_vcf = os.path.join(out_dir, f'{source}.vcf.gz')
            self._concat_vcfs(vcfs, out_vcf)

            if unrelated_samples_only:
                sampleInfo_file = os.path.join(out_dir, f'{source}_unrelated_sampleInfo.txt')
                sampleID_file = sampleInfo_file.replace('.txt', '_sampleID.txt')
                self._get_unrelated_samples(sampleInfo_file, sampleID_file)
                extracted_file = os.path.join(out_dir, f'{source}_unrelated.vcf.gz')
                self._extract_samples(out_vcf, sampleID_file, extracted_file)

    def merge_dataset_with_reference(self, dataset_vcf, reference_vcf, out_file, threads=1):
        try:
            dataset_bed = dataset_vcf.split('.vcf')[0]
            reference_bed = reference_vcf.split('.vcf')[0]

            if os.path.exists(dataset_bed + '.bed') and os.path.exists(dataset_bed + '.bim') and os.path.exists(dataset_bed + '.fam'):
                print(f"{dataset_bed} bed/bim/fam already exist. Skipping conversion from vcf.")
            else:
                cmd = f'plink --keep-allele-order --vcf {dataset_vcf} --make-bed --out {dataset_bed} --threads {threads}'
                subprocess.run(cmd, shell=True, check=True)

            if os.path.exists(reference_bed + '.bed') and os.path.exists(reference_bed + '.bim') and os.path.exists(reference_bed + '.fam'):
                print(f"{reference_bed} bed/bim/fam already exist. Skipping conversion from vcf.")
            else:
                cmd = f'plink --keep-allele-order --vcf {reference_vcf} --make-bed --out {reference_bed} --threads {threads}'
                subprocess.run(cmd, shell=True, check=True)

            if not os.path.exists(dataset_bed + '.bim') or not os.path.exists(reference_bed + '.bim'):
                print("bed/bim/fam files not found for dataset or reference. Cannot proceed with merging.")
                return

            shared_variants_file = dataset_bed + '_shared_variants.txt'
            self._get_shared_variants(dataset_bed + '.bim', reference_bed + '.bim', shared_variants_file, update_bim=True)

            dataset_extracted_bed = dataset_bed + '_extracted'
            reference_extracted_bed = reference_bed + '_extracted'

            cmd = f'plink --keep-allele-order --bfile {dataset_bed} --extract {shared_variants_file} --make-bed --out {dataset_extracted_bed}'
            subprocess.run(cmd, shell=True, check=True)
            cmd = f'plink --keep-allele-order --bfile {reference_bed} --extract {shared_variants_file} --make-bed --out {reference_extracted_bed}'
            subprocess.run(cmd, shell=True, check=True)

            cmd = f'plink --keep-allele-order --bfile {reference_extracted_bed} --bmerge {dataset_extracted_bed} --make-bed --out {out_file} --threads {threads}'
            subprocess.run(cmd, shell=True, check=True)
        except Exception as e:
            print(f"Error merging VCFs: {e}")

    def _get_shared_variants(self, dataset_bim, reference_bim, out_file, update_bim=True):
        try:
            df1 = pd.read_table(dataset_bim, header=None, sep='\t')
            df1.iloc[:, 1] = df1.apply(lambda row: ':'.join([str(row[0]), str(row[3]), str(row[5]), str(row[4])]), axis=1)
            df2 = pd.read_table(reference_bim, header=None, sep='\t')
            df2.iloc[:, 1] = df2.apply(lambda row: ':'.join([str(row[0]), str(row[3]), str(row[5]), str(row[4])]), axis=1)
            shared_variants = set(df1.iloc[:, 1]).intersection(set(df2.iloc[:, 1])) 
            with open(out_file, 'w') as f:
                for n in range(df2.shape[0]):
                    k = df2.iloc[n, 1]
                    if k in shared_variants:
                        f.write(k + '\n')
            print(f"Shared variants saved to {out_file}")
            if update_bim:
                df1.to_csv(dataset_bim, header=False, index=False, sep='\t')
                df2.to_csv(reference_bim, header=False, index=False, sep='\t')
        except Exception as e:
            print(f"Error getting shared variants: {e}")

    def feature_engineering(self, in_file, out_file='data/features.txt', pruning=True, pruning_params=[50, 5, 0.2], pca=True, pca_params=[50], threads=1):
        out_file_bed = f'{in_file}_pruned'
        out_file_pca = f'{in_file}_pca'
        try:
            if pruning:
                cmd = f'plink --keep-allele-order --bfile {in_file} --indep-pairwise {pruning_params[0]} {pruning_params[1]} {pruning_params[2]} --out {in_file} --threads {threads}'
                subprocess.run(cmd, shell=True, check=True)
                cmd = f'plink --keep-allele-order --bfile {in_file} --extract {in_file}.prune.in --make-bed --out {out_file_bed} --threads {threads}'
                subprocess.run(cmd, shell=True, check=True)
                infile = out_file_bed

            if pca:
                cmd = f'plink --bfile {in_file} --pca {pca_params[0]} --out {out_file_pca} --threads {threads}'
                subprocess.run(cmd, shell=True, check=True)
                print(f"PCA results saved to {out_file_pca}.eigenvec")

                df = pd.read_table(f'{out_file_pca}.eigenvec', header=None, sep=' ')
                df.columns = ['FID', 'IID'] + [f'PC{i}' for i in range(1, df.shape[1]-1)]
                df.to_csv(out_file, header=True, index=False, sep='\t')
                print(f"Features saved to {out_file}")
            else:
                print('PCA is the only feature engineering method implemented currently.')
                return
        except Exception as e:
            print(f"Error in feature engineering: {e}")

    def add_labels(self, in_file='data/features.txt', label_file='data/1000genomes_unrelated_sampleInfo.txt'):
        D = {}
        df_labels = pd.read_table(label_file, header=0, sep='\t')
        for n in range(df_labels.shape[0]):
            k = df_labels['SampleID'].iloc[n]
            D[k] = [df_labels['Population'].iloc[n], df_labels['Superpopulation'].iloc[n]]

        df = pd.read_table(in_file, header=0, sep='\t')
        df.columns = ['FID', 'SampleID'] + list(df.columns)[2:]
        dfp = df.iloc[:, 1:].copy()
        dfsp = df.iloc[:, 1:].copy()
        dfp['class_name'] = [D[k][0] if k in D else '.' for k in dfp['SampleID']]
        dfsp['class_name'] = [D[k][1] if k in D else '.' for k in dfsp['SampleID']]

        dfp_labeled = dfp[dfp['class_name'] != '.'].copy()
        dfsp_labeled = dfsp[dfsp['class_name'] != '.'].copy()
        dfsp_unlabeled = dfsp[dfsp['class_name'] == '.'].copy()
        dfsp_unlabeled.drop(columns=['class_name'], inplace=True)

        out_file_population = in_file.replace('.txt', '_labels_Population.txt')
        out_file_superpopulation = in_file.replace('.txt', '_labels_Superpopulation.txt')
        p_labels = sorted(dfp_labeled['class_name'].unique())
        sp_labels = sorted(dfsp_labeled['class_name'].unique())
        df_p_labels = pd.DataFrame({'class_name': p_labels, 'class': range(len(p_labels))})
        df_sp_labels = pd.DataFrame({'class_name': sp_labels, 'class': range(len(sp_labels))})
        df_p_labels.to_csv(out_file_population, header=True, index=False, sep='\t')
        df_sp_labels.to_csv(out_file_superpopulation, header=True, index=False, sep='\t')
        print(f"Population labels saved to {out_file_population}")
        print(f"Superpopulation labels saved to {out_file_superpopulation}")

        dfp_labeled['class'] = [p_labels.index(x) for x in dfp_labeled['class_name']]
        dfsp_labeled['class'] = [sp_labels.index(x) for x in dfsp_labeled['class_name']]

        dfp_labeled.to_csv(in_file.replace('.txt', '_labeled_Population.txt'), header=True, index=False, sep='\t')
        dfsp_labeled.to_csv(in_file.replace('.txt', '_labeled_Superpopulation.txt'), header=True, index=False, sep='\t')
        dfsp_unlabeled.to_csv(in_file.replace('.txt', '_ToPred.txt'), header=True, index=False, sep='\t')
        print(f"Labeled Population data saved to {in_file.replace('.txt', '_labeled_Population.txt')}")
        print(f"Labeled Superpopulation data saved to {in_file.replace('.txt', '_labeled_Superpopulation.txt')}")
        print(f"Unlabeled data saved to {in_file.replace('.txt', '_ToPred.txt')}")

    def split_train_test(self, in_file, test_size=0.2, random_state=42):
        df = pd.read_table(in_file, header=0, sep='\t')
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['class'])
        df_train.to_csv(in_file.replace('.txt', '_train.txt'), header=True, index=False, sep='\t')
        df_test.to_csv(in_file.replace('.txt', '_test.txt'), header=True, index=False, sep='\t')
        print(f"Train data saved to {in_file.replace('.txt', '_train.txt')}")
        print(f"Test data saved to {in_file.replace('.txt', '_test.txt')}")

    def _concat_vcfs(self, vcf_list, out_vcf):
        try:
            cmd = ['bcftools', 'concat', '-Oz', '-o', out_vcf] + vcf_list
            subprocess.run(cmd, check=True)
            subprocess.run(['bcftools', 'index', out_vcf], check=True)
            print(f"Combined VCF saved to {out_vcf}")
        except Exception as e:
            print(f"Error concatenating VCFs: {e}")

    def _get_unrelated_samples(self, out_file, out_file2):
        in_file1 = self.url_1000genomes_sample_populatioin
        in_file2 = self.url_1000genomes_sample_unrelated

        df1 = pd.read_table(in_file1, header=0, sep=' ')
        df2 = pd.read_table(in_file2, comment='#', header=None)

        df1_sub = df1.loc[df1['SampleID'].isin(df2[9].str.strip())]
        df1_sub.to_csv(out_file, header=True, index=False, sep='\t')
        df1_sub['SampleID'].to_csv(out_file2, header=False, index=False, sep='\t')

    def _extract_samples(self, vcf_file, sample_file, out_file):
        try:
            print(f'Extracting unrelated samples')
            cmd = ['bcftools', 'view', '-S', sample_file, '-Oz', '-o', out_file, vcf_file]
            subprocess.run(cmd, check=True)
            subprocess.run(['bcftools', 'index', out_file], check=True)
            print(f"Extracted samples VCF saved to {out_file}")
        except Exception as e:
            print(f"Error extracting samples: {e}")

if __name__ == '__main__':
    an = Ancestry()
    #an.get_ancestry_reference()
    #an.merge_dataset_with_reference('data/Omni.vcf.gz', 'data/1000genomes_unrelated.vcf.gz', 'data/merged')
    #an.feature_engineering('data/merged', pruning=True, pca=True)
    #an.add_labels()
    #an.split_train_test()
