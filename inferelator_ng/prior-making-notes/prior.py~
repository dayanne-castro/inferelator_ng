import pickle
import pysam
import pybedtools
import argparse
import pandas
import statistics as stat
from multiprocessing import Pool
from sklearn.cluster import KMeans

def pooler(arg, **kwarg):
    class_reference = arg[0]
    paramter = arg[1]
    class_function = arg[2]
    function_to_pool = getattr(class_reference,class_function)
    return function_to_pool(paramter)

class Motifs:
    """Motifs main class.

    -----------------
    Attributes
    -----------------

    bam_file: alignment file in BAM format containing reads from an ATAC-seq
    experiment; file should be sorted and indexed.

    motifs_file: transcription factor motifs in open regions of the genome
    formatted as:
                 - chromosome start end motif-ID score strand
                   (score can be e-value, p-value, doesn't matter...)
                 - tab delimited, no header,
                   should include only motifs in peaks

    motifs: holds a dictionary with motifs from motifs_file;
            ---
            {TF1:[(occurence1_chrom, occurence1_start, occurence1_end, occurence1_p-value, occurence1_strand),
                  (occurence2_chrom, occurence2_start, occurence2_end, occurence2_p-value, occurence2_strand)],
             TF2:[(occurence1_chrom, occurence1_start, occurence1_end, occurence1_p-value, occurence1_strand)]...}
            ---
            Keys: motif IDs -- will be used later to build the prior, thus I use transcription factors as IDs.
            Values: list of occurences in the genome of the motif(s) associated with identifier.

    insertion_counts: holds a dictionary with matrices of insertion counts around motifs for a particular transcription factor
                      ---
                      Keys: motif holders -- will be used later to build the prior
                      Values: pandas DataFrame with insertion counts around each motif for the motif(s) associated with identifier,
                              rownames are motif holders -- chrom,start,end,tf,score,strand

    tn5_9mer_counts: holds a dictionary with matrices of 9mer counts around motifs for a particular transcription factor
                     ---
                     Keys: motif holders -- will be used later to build the prior
                     Values: pandas DataFrame with 9mer counts around each motif for the motif(s) associated with identifier,
                             rownames are motif holders as above

    motif_clusters_insertion_counts: hold a dictionary with clusters of motifs to 'keep' or 'not_keep' -- generated using insertion counts
                                     Keys: motif IDs (as above)
                                     Values: pandas DataFrame with clusters of motifs associated with identifier.
                                     rownames are motif holders as above

    motif_clusters_9mer_counts: hold a dictionary with clusters of motifs to 'keep' or 'not_keep' -- generated using 9mer counts
                                     Keys: motif IDs (as above)
                                     Values: pandas DataFrame with clusters of motifs associated with identifier.
                                     rownames are motif holders as above

    """

    def __init__(self, bam_file, motifs_file):

        self.bam_file = bam_file
        self.motifs_file = motifs_file
        self.motifs = {}
        self.insertion_counts = {}
        self.tn5_9mer_counts = {}
        self.motif_clusters_insertion_counts = {}
        self.motif_clusters_9mer_counts = {}
        self.process_count = 4

    def read_motifs_file(self):

        """
        Reads in a motifs file and saves a dictionary of motifs for each transcription factor in self.motifs
        """

        motifs_file = self.motifs_file

        with open(motifs_file, "r") as motifs_handle:

            for motif_occurence in motifs_handle:

                motif_occurence = motif_occurence.split()

                motif_id = motif_occurence[3]
                chrom = motif_occurence[0]
                start = motif_occurence[1]
                end = motif_occurence[2]
                strand = motif_occurence[5]
                score = motif_occurence[4]

                to_bed = [chrom, int(start), int(end), motif_id, score, strand]

                if motif_id in self.motifs:
                    self.motifs[motif_id].append(tuple(to_bed))
                else:
                    self.motifs[motif_id] = [tuple(to_bed)]

    def get_insertions(self, chrom, start, end, upstream=50, downstream=50):

        """
        Returns insertion counts at a given region of the genome
        """

        bam_handle = pysam.AlignmentFile(self.bam_file, "rb")

        # initialize counts vector at 0 for all positions
        region_length = int((end - start))
        insertion_counts = [0] * region_length

        # fetch reads mapping to the region specified as input
        reads = bam_handle.fetch(chrom, start, end)

        # each read represents a potential insertion within region
        for read in reads:

            if read.is_reverse:
                # offset by 5 bp
                insertion_pos = read.reference_end - 5
            else:
                # offset by 4 bp
                insertion_pos = read.reference_start + 4

            pos = insertion_pos - start

            # make sure pos is within region
            if pos in range(0, region_length):
                insertion_counts[pos] += 1

        return tuple(insertion_counts)

    @staticmethod
    def get_tn5_9mer(insertion_counts, up_offset=4, down_offset=5):

        """
        Smooth insertion counts to Tn5 occupancy (9mer track)
        """

        tn5_9mer_counts = list(insertion_counts)

        region = range(0, len(insertion_counts))

        for pos in region:
            for idx in range(pos - up_offset, pos + down_offset):
                if idx in region and idx != pos:
                    tn5_9mer_counts[idx] += insertion_counts[pos]

        return tuple(tn5_9mer_counts)

    def compute_scores_matrices(self, upstream, downstream):

        """
        Generate matrices of insertion and 9mer counts for all motifs in self.motifs
        """

        for motif_id in self.motifs.keys():  # Paralelize

            insertion_counts_mat = []
            tn5_9mer_counts_mat = []
            rownames_motifs = []

            motifs = pybedtools.BedTool(self.motifs[motif_id])
            motifs = motifs.sort()

            for motif in motifs:

                # get motif coordinates -- FIMO output is a closed interval, so add 1 to end (python)
                chrom = str(motif.chrom)
                center = round(stat.median(range(motif.start, motif.end)))
                start = int(center - upstream)
                end = int(center + downstream)

                insertion_counts = self.get_insertions(chrom, start, end)
                insertion_counts_mat.append(insertion_counts)

                tn5_9mer_counts = Motifs.get_tn5_9mer(insertion_counts)
                tn5_9mer_counts_mat.append(tn5_9mer_counts)

                motif_holder = ",".join([motif.chrom, str(motif.start), str(motif.end), motif.name, motif.score, motif.strand])
                rownames_motifs.append(motif_holder)

            insertion_counts_df = pandas.DataFrame(insertion_counts_mat,
                                                   index=rownames_motifs)

            tn5_9mer_counts_df = pandas.DataFrame(tn5_9mer_counts_mat,
                                                  index=rownames_motifs)

            self.insertion_counts[motif_id] = insertion_counts_df
            self.tn5_9mer_counts[motif_id] = tn5_9mer_counts_df

    def pool_compute_scores_matrices(self, motif_id):
        """
        Generate matrices of insertion and 9mer counts for all motifs in self.motifs
        """
        upstream = 50
        downstream = 50
        insertion_counts_mat = []
        tn5_9mer_counts_mat = []
        rownames_motifs = []

        motifs = pybedtools.BedTool(self.motifs[motif_id])
        motifs = motifs.sort()

        for motif in motifs:

            # get motif coordinates -- FIMO output is a closed interval, so add 1 to end (python)
            chrom = str(motif.chrom)
            center = round(stat.median(range(motif.start, motif.end)))
            start = int(center - upstream)
            end = int(center + downstream)

            insertion_counts = self.get_insertions(chrom, start, end)
            insertion_counts_mat.append(insertion_counts)

            tn5_9mer_counts = Motifs.get_tn5_9mer(insertion_counts)
            tn5_9mer_counts_mat.append(tn5_9mer_counts)

            motif_holder = ",".join([motif.chrom, str(motif.start), str(motif.end), motif.name, motif.score, motif.strand])
            rownames_motifs.append(motif_holder)

        insertion_counts_df = pandas.DataFrame(insertion_counts_mat, index=rownames_motifs)
        tn5_9mer_counts_df = pandas.DataFrame(tn5_9mer_counts_mat, index=rownames_motifs)

        return {"motif_id": motif_id, "insertion_counts": insertion_counts_df, "tn5_9mer_counts": tn5_9mer_counts_df}

    def threaded_compute_scores_matrices(self):
        matrix_pool = Pool(processes=self.process_count)
        args = []

        for motif_id in self.motifs.keys():
            args.append([self,motif_id,"pool_compute_scores_matrices"])
        results = matrix_pool.map(pooler, args)
        matrix_pool.close()
        matrix_pool.join()

        for data in results:
            self.insertion_counts[data["motif_id"]] = data["insertion_counts"]
            self.tn5_9mer_counts[data["motif_id"]] = data["tn5_9mer_counts"]

    def pool_clusters_motifs(self, motif_id):
        insertion_counts_df = self.insertion_counts[motif_id]
        insertion_counts_mat = insertion_counts_df.as_matrix()

        motif_clusters = KMeans(n_clusters=2)
        motif_clusters.fit(insertion_counts_mat)
        labels = motif_clusters.labels_
        results = pandas.DataFrame([insertion_counts_df.index, labels]).T
        results.columns = ['motif', 'cluster']

        clust0 = insertion_counts_df[insertion_counts_df.index.isin(results['motif'][results['cluster'] == 0])]
        clust1 = insertion_counts_df[insertion_counts_df.index.isin(results['motif'][results['cluster'] == 1])]

        clust0_mean = clust0.mean(1).mean()
        clust1_mean = clust1.mean(1).mean()

        if clust0_mean > clust1_mean:

            results['cluster'][results['cluster'] == 0] = 'keep'
            results['cluster'][results['cluster'] == 1] = 'not_keep'

        if clust1_mean > clust0_mean:

            results['cluster'][results['cluster'] == 1] = 'keep'
            results['cluster'][results['cluster'] == 0] = 'not_keep'

        return_result = {"motif_id": motif_id, "insertion_counts": results,"9mer_counts": results}

        tn5_9mer_counts_df = self.tn5_9mer_counts[motif_id]
        tn5_9mer_counts_mat = tn5_9mer_counts_df.as_matrix()

        motif_clusters = KMeans(n_clusters=2)
        motif_clusters.fit(tn5_9mer_counts_mat)
        labels = motif_clusters.labels_
        results = pandas.DataFrame([tn5_9mer_counts_df.index, labels]).T
        results.columns = ['motif', 'cluster']

        clust0 = insertion_counts_df[insertion_counts_df.index.isin(results['motif'][results['cluster'] == 0])]
        clust1 = insertion_counts_df[insertion_counts_df.index.isin(results['motif'][results['cluster'] == 1])]

        clust0_mean = clust0.mean(1).mean()
        clust1_mean = clust1.mean(1).mean()

        if clust0_mean > clust1_mean:
            results['cluster'][results['cluster'] == 0] = 'keep'
            results['cluster'][results['cluster'] == 1] = 'not_keep'

        if clust1_mean > clust0_mean:
            results['cluster'][results['cluster'] == 1] = 'keep'
            results['cluster'][results['cluster'] == 0] = 'not_keep'

        return_result["9mer_counts"] = results
        return return_result

    def threaded_cluster_motifs(self):

        """
        Use K-Means to cluster motifs for a given transcription factor in two clusters for all TFs in self.motifs
        """
        #  Paralelize, as of now I am going to do both ins only
        # insertion counts

        motif_pool = Pool()
        # motif_pool.map(pool_clusters_motifs_insertion_counts, self.insertion_counts.keys())
        args = []
        for x in self.insertion_counts.keys():
            args.append([self, x, "pool_clusters_motifs"])
        results = motif_pool.map(pooler, args)
        motif_pool.close()
        motif_pool.join()
        for data in results:
            self.motif_clusters_insertion_counts[data["motif_id"]] = data["insertion_counts"]
            self.motif_clusters_9mer_counts[data["motif_id"]] = data["9mer_counts"]


    def cluster_motifs(self):

        """
        Use K-Means to cluster motifs for a given transcription factor in two clusters for all TFs in self.motifs
        """
        #  Paralelize, as of now I am going to do both ins and 9mer (choose one later?)

        # insertion counts
        for motif_id in self.insertion_counts.keys():

            insertion_counts_df = self.insertion_counts[motif_id]
            insertion_counts_mat = insertion_counts_df.as_matrix()

            motif_clusters = KMeans(n_clusters=2)
            motif_clusters.fit(insertion_counts_mat)
            labels = motif_clusters.labels_
            results = pandas.DataFrame([insertion_counts_df.index, labels]).T
            results.columns = ['motif', 'cluster']

            clust0 = insertion_counts_df[insertion_counts_df.index.isin(results['motif'][results['cluster'] == 0])]
            clust1 = insertion_counts_df[insertion_counts_df.index.isin(results['motif'][results['cluster'] == 1])]

            clust0_mean = clust0.mean(1).mean()
            clust1_mean = clust1.mean(1).mean()

            if clust0_mean > clust1_mean:

                results['cluster'][results['cluster'] == 0] = 'keep'
                results['cluster'][results['cluster'] == 1] = 'not_keep'

            if clust1_mean > clust0_mean:

                results['cluster'][results['cluster'] == 1] = 'keep'
                results['cluster'][results['cluster'] == 0] = 'not_keep'

            self.motif_clusters_insertion_counts[motif_id] = results

        # 9mers
        for motif_id in self.tn5_9mer_counts.keys():

            tn5_9mer_counts_df = self.tn5_9mer_counts[motif_id]
            tn5_9mer_counts_mat = tn5_9mer_counts_df.as_matrix()

            motif_clusters = KMeans(n_clusters=2)
            motif_clusters.fit(tn5_9mer_counts_mat)
            labels = motif_clusters.labels_
            results = pandas.DataFrame([tn5_9mer_counts_df.index, labels]).T
            results.columns = ['motif', 'cluster']

            clust0 = insertion_counts_df[insertion_counts_df.index.isin(results['motif'][results['cluster'] == 0])]
            clust1 = insertion_counts_df[insertion_counts_df.index.isin(results['motif'][results['cluster'] == 1])]

            clust0_mean = clust0.mean(1).mean()
            clust1_mean = clust1.mean(1).mean()

            if clust0_mean > clust1_mean:

                results['cluster'][results['cluster'] == 0] = 'keep'
                results['cluster'][results['cluster'] == 1] = 'not_keep'

            if clust1_mean > clust0_mean:

                results['cluster'][results['cluster'] == 1] = 'keep'
                results['cluster'][results['cluster'] == 0] = 'not_keep'

            self.motif_clusters_9mer_counts[motif_id] = results

class Edge:

    """
    Container for edges in the prior
    ----------
    Attributes
    ----------
    regulator: transcription factor name
    target: target gene name
    weight: weight to be put in the prior matrix
    motif: motif coordinates
    gene: gene coordinates -- obs. NOW I HAVE THE TSS HERE, SHOULD BE CHANGED
    distance: peak distance to gene
    """

    def __init__(self, regulator, target, weight, motif, gene, distance, peak):

        self.regulator = regulator
        self.target = target
        self.weight = weight
        self.motif = motif
        self.gene = gene
        self.distance = distance
        self.peak = peak

class Prior:

    """
    ----------------
    Attributes
    -----------------

    bam_file: aligned ATAC-seq BAM file

    motifs_file: motifs annotation in FIMO format IN PEAKS

    peaks_file: ATAC peaks in BED format (e.g. called by MACS)

    tss_file: TSS annotation in BED format

    transcription_factors:

    target_genes:

    upstream_in_matrix:

    downstream_in_matrix:

    cluster: whether to cluster motifs or not

    max_upstream_distance: bp upstream allowed for assignment peak to gene

    max_downstream_distance: bp downstream allowed for assignment peak to gene

    """

    def __init__(self, bam_file, motifs_file, peaks_file, tss_file,
                 transcription_factors, target_genes, cluster=True,
                 upstream_in_matrix=50, downstream_in_matrix=50,
                 max_upstream_distance=float("Inf"),
                 max_downstream_distance=float("Inf")):

        self.bam_file = bam_file
        self.motifs_file = motifs_file
        self.peaks_file = peaks_file
        self.tss_file = tss_file
        self.transcription_factors = transcription_factors
        self.target_genes = target_genes
        self.upstream_in_matrix = upstream_in_matrix
        self.downstream_in_matrix = downstream_in_matrix
        self.cluster = cluster
        self.max_upstream_distance = max_upstream_distance
        self.max_downstream_distance = max_downstream_distance

        self.motifs = []
        self.peaks = []
        self.tss = []
        self.edges = []

    def __str__(self):

        print("Prior generated from: ")
        print("Alignment file: {}".format(self.bam_file))
        print("Motifs file: {}".format(self.motifs_file))
        print("Peaks file: {}".format(self.peaks_file))
        print("TSS annotation file: {}".format(self.tss_file))
        #  add other parameters later

    def get_motifs(self):
        """
        returns a BedTool with good motifs: all if no clustering, else motifs in cluster 'keep'.
        """

        if self.cluster is True:

            motifs_keep_ins = []
            #  motifs_keep_9mer = []

            #  class Motifs to cluster etc
            motifs = Motifs(self.bam_file, self.motifs_file)
            motifs.read_motifs_file()
            # motifs.compute_scores_matrices(upstream=self.upstream_in_matrix,
            #                                downstream=self.downstream_in_matrix)
            motifs.threaded_compute_scores_matrices()
            # print(a)
            # motifs.cluster_motifs()
            motifs.threaded_cluster_motifs()

            # find out which motifs to keep
            for regulator in motifs.motif_clusters_insertion_counts.keys():

                keep = motifs.motif_clusters_insertion_counts[regulator]['cluster'] == 'keep'
                tmp_motifs_keep = motifs.motif_clusters_insertion_counts[regulator]['motif'][keep]

                # append good motifs
                for motif in tmp_motifs_keep:
                    motifs_keep_ins.append(tuple(motif.split(',')))

            # 9mer -- after will keep only one?
            # for regulator in motifs.motif_clusters_9mer_counts.keys():

            #    keep = motifs.motif_clusters_9mer_counts[regulator]['cluster'] == 'keep'
            #    tmp_motifs_keep = motifs.motif_clusters_9mer_counts[regulator]['motif'][keep]

                # append good motifs
            #    for motif in motifs_keep:
            #        motifs_keep_9mer.append(tuple(motif.split(',')))

            motifs = pybedtools.BedTool(motifs_keep_ins)
            #  motifs = pybedtools.BedTool(motifs_keep_9mer)

        else:

            motifs = pybedtools.BedTool(self.motifs_file)

        self.motifs = motifs.sort()

    def get_peaks(self):

        """
        returns a BedTool with peaks in peaks_file
        """

        peaks = pybedtools.BedTool(self.peaks_file)
        peaks = peaks.sort()

        self.peaks = peaks

    def get_tss(self):

        """
        returns a BedTool with TSS coordinates in tss_file
        """

        tss_genes = pybedtools.BedTool(self.tss_file)
        tss_genes = tss_genes.sort()

        self.tss = tss_genes

    def assign_motifs_to_genes(self):

        """assign motifs to genes -- first assign peaks to genes and then all
        motifs in that peak to that gene. returns a BedTool with motif - peak
        - gene coordinates
        """
        motifs = self.motifs
        peaks = self.peaks
        tss = self.tss

        peaks_to_genes = peaks.closest(tss, D='b')
        motifs_to_genes = motifs.closest(peaks_to_genes.sort())

        return(motifs_to_genes)

    def get_edges(self):

        """

        """

        motifs_to_genes = self.assign_motifs_to_genes()

        for assignment in motifs_to_genes:

            assignment = assignment.fields
            regulators = assignment[3].split('::')
            target = assignment[18].split()[0]
            distance = int(assignment[21])
            peak = assignment[9]
            weight = 1

            motif = (assignment[0].encode('ascii', 'ignore'),
                     assignment[1].encode('ascii', 'ignore'),
                     assignment[2].encode('ascii', 'ignore'),
                     assignment[3].encode('ascii', 'ignore'))
            gene = (assignment[15].encode('ascii', 'ignore'),
                    assignment[16].encode('ascii', 'ignore'),
                    assignment[17].encode('ascii', 'ignore'),
                    assignment[18].encode('ascii', 'ignore'))

            for regulator in regulators:

                edge = Edge(regulator.encode('ascii', 'ignore'),
                            target.encode('ascii', 'ignore'),
                            weight,
                            motif,
                            gene,
                            distance,
                            peak.encode('ascii', 'ignore'))

                self.edges.append(edge)

    def make_prior(self):
        """call all functions to generate the prior and fill in edges attribute
        """

        self.get_motifs()
        self.get_peaks()
        self.get_tss()
        self.get_edges()

    def make_prior_matrix(self):
        """ returns a pandas DataFrame with the prior matrix using
        all edges in self.edges rows are genes in self.target_genes,
        columns are the TFs in self.transcription_factors
        """

        prior = pandas.DataFrame(0, index=self.target_genes,
                                 columns=self.transcription_factors)

        for edge in self.edges:

            if edge.target in self.target_genes and edge.regulator in self.transcription_factors:

                prior.ix[edge.target, edge.regulator] = edge.weight

        return(prior)




def main(args):

    motifs_file = args['bed_file']
    bam_file = args['bam_file']
    peaks_file = args['peaks_file']
    tss_file = args['tss_file']
    transcription_factors_file = args['transcription_factors_file']
    target_genes_file =  args['target_genes_file']
    output_preffix = args['output_preffix']

    transcription_factors = []
    with open(transcription_factors_file, "r") as tfs_handle:
        for tf in tfs_handle:
            transcription_factors.append(tf.split()[0])

    target_genes = []
    with open(target_genes_file, "r") as targets_handle:
        for gene in targets_handle:
            target_genes.append(gene.split()[0])

    motifs_file = './input-tests/motifs_test_file.bed'
    bam_file = './input-tests/bam_test_file.bam'
    peaks_file = './input-tests/peaks_test_file.bed'
    tss_file = './input-tests/tsses_test_file.bed'
    transcription_factors = ['TF1', 'TF2', 'TF3']
    target_genes = ['gene1', 'gene2']

    prior = Prior(bam_file, motifs_file,
                  peaks_file, tss_file,
                  transcription_factors, target_genes,
                  cluster=True, upstream_in_matrix=50, downstream_in_matrix=50,
                  max_upstream_distance=float("Inf"), max_downstream_distance=float("Inf"))

    prior.make_prior()

    mat = prior.make_prior_matrix()
    mat.to_csv('_'.join([output_preffix, 'mat.tsv']), sep = '\t')

    pickle.dump(prior, open('_'.join([output_preffix, 'priorObj.p']), "wb"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'generate a prior on network structure based on Tn5 insertion track')
    parser.add_argument('-bed','--bed_file', help = 'bed file with motifs in open regions in chromatin', required = True)
    parser.add_argument('-bam','--bam_file', help = 'BAM file with ATAC-seq alignments', required = True)
    parser.add_argument('-pk', '--peaks_file', help = 'ATACseq peaks', required = True)
    parser.add_argument('-tss','--tss_file', help = 'bigWig file with Tn5 insertion frequencies', required = True)
    #parser.add_argument('-u','--max_upstream_distance', help = 'pick motifs at most "u" bp upstream from gene TSS; default = Inf', required = False, default = float('inf'))
    #parser.add_argument('-d','--max_downstream_distance', help = 'pick motifs at most "d" bp upstream from gene TSS; default = Inf', required = False, default = float('inf'))
    #parser.add_argument('-c','--cores', help = 'number of cores; default = 1', required = False, default = 1)
    #parser.add_argument('-o','--output_directory', help = 'output directory where to save bed, heatmaps and prior files; default = ./output', required = False, default = './output')
    parser.add_argument('-tfs','--transcription_factors_file', help = 'file with transcription factors of interest, one by line', required = True)
    parser.add_argument('-tgs','--target_genes_file', help = 'file with target genes of interest, one by line', required = True)
    parser.add_argument('-o','--output_preffix', help = 'output_preffix -- e.g. path/to/output/prior1', required = True)
    args = vars(parser.parse_args())
    main(args)


# call
# python prior.py \
# -bed ./input-tests/motifs_test_file.bed \
# -bam ./input-tests/bam_test_file.bam \
# -pk ./input-tests/peaks_test_file.bed \
# -tss ./input-tests/tsses_test_file.bed \
# -tgs ./input-tests/target_genes_test_file.txt \
# -tfs ./input-tests/tfs_test_file.txt \
# -o output_python_script_test
