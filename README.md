# DirectHRD

DirectHRD is an ultrasensitive scar-based classifier to detect HRD from low tumor fraction samples such as liquid biopsies using whole-genome sequencing (WGS). DirectHRD encompasses two components which are small Indel calling and HRD calling. This workspace contains code only for the HRD calling part. In theory, a user can use their favorite Indel caller for WGS data. However, we do recommend using CODECsuite for Indel calling. 

# To install DirectHRD
  1. `git clone https://github.com/broadinstitute/DirectHRD.git`
  2. `cd DirectHRD && pip install .`

# To use the Recommended Indel caller
CODECsuite is available here at https://github.com/broadinstitute/CODECsuite. For installation, please refer to that github page. The command line for running is as follow: 

        CODECsuite/build/codec call -b ~{tumor_or_ctdna_bam} \
            -r ~{reference_fasta} \
            -L ~{eval_genome_bed} \
            -n ~{germline_bam} \
            -V ~{population_based_vcf} \
            -m 60 \
            -q 30 \
            -d 12 \
            -x 6 \
            -c 3 \
            -5 \
            -g 30 \
            -Q 0.5 \
            -B 0.6 \
            -Y 10 \
            -W 1 \
            -f 30 \
            -E 8 \
            -s \
            -I 1 \
            -R 1 \
            -u \
            -i ~{max_allele_frac} \
            -o ~{sample_id}
The three required input arguments are: 

`tumor_or_ctdna_bam`: the sample that the HRD status is investiaged. 

`reference_fasta`: In the paper, we use GRCh37 reference genome. 

`eval_genome_bed`: a bed file contains regions under investigation. We recommand using [GRCh37 high complexitiy regions](https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/genome-stratifications/v3.0/GRCh37/LowComplexity/GRCh37_notinAllTandemRepeatsandHomopolymers_slop5.bed.gz)

The `germline_bam` is highly recommand to have for the purpose of calling tumor-specific mutations but in the case that this is not avaialbe, user can omit this option. 

The `population_based_vcf` is another good to have input file which can mitigate contamination and low germline depth. We recommand using the ALFA dataset from dbsnp https://www.ncbi.nlm.nih.gov/snp/docs/gsr/alfa/.  

# HRD calling

In practice, a user can use any Indel caller such as Mutect2 or Strelka2 to call Indels. However, I do recommend post-filtering the Indel calls using a low comlexity filter such as genome in a bottle [GRCh37 high complexitiy regions](https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/genome-stratifications/v3.0/GRCh37/LowComplexity/GRCh37_notinAllTandemRepeatsandHomopolymers_slop5.bed.gz). 

The first step of HRD calling invovling Indel classification to COSMIC ID83 format. We used COSMIC v3.2 and [SigProfiler](https://cancer.sanger.ac.uk/signatures/tools/) in the paper. The second step is the HRDscore prediction using a Multinomial Mixture Model (MMM).

To use the pacakge, run: 

`hrd-classifier indel_vcfs_folder -p project_name -o output.tsv`


