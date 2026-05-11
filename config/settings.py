import os
from pathlib import Path
from pydantic import BaseModel
from typing import Literal

INITIAL = """
Given the predefined pipeline steps below, select exactly ONE tool for each step.

Follow these rules strictly:
- Each tool must be a specific, standalone tool (not a toolkit, suite, or ecosystem)
- Do not output meta-tools or multi-function frameworks (e.g. SAMtools, GATK, Bioconductor)
- Do not include QC tools (e.g. FastQC, Trimmomatic, fastp, MultiQC)
- Select the tool that is most appropriate for the specific biological and technical context described
"""


TASKS = {
    "genome_assembly":
        "Select exactly one tool for each step in the following genome assembly pipeline "
        "targeting a small eukaryotic genome (~30 Mb) with high heterozygosity:\n"
        "1. K-mer-based error correction of short paired-end reads prior to assembly\n"
        "2. De novo assembly from short-read Illumina data optimized for heterozygous genomes\n",

    "variant_detection":
        "Select exactly one tool for each step in the following structural variant "
        "detection pipeline for a tumor-normal paired sample:\n"
        "1. Short-read alignment with sensitivity to novel junction sites in somatic tissue\n"
        "2. Copy number alteration detection from read depth in whole exome sequencing data\n"
        "3. Deletion and inversion detection from paired-end discordant reads\n",

    "RNA-seq":
        "Select exactly one tool for each step in the following transcriptome "
        "analysis pipeline for a plant RNA-seq experiment:\n"
        "1. Splice-aware alignment of RNA-seq reads to a reference genome\n"
        "2. Differential alternative splicing analysis comparing two conditions\n"
        "3. Chimeric transcript identification from aligned RNA-seq reads\n"
        "4. Differential expression analysis in the context of alternative splicing events\n",
    
    "Phylogeny":
        "Select exactly one tool for each step in the following phylogenetic "
        "analysis pipeline for a dataset of ~500 distantly related protein sequences:\n"
        "1. Iterative profile search to identify homologous sequences from a protein database\n"
        "2. Progressive multiple sequence alignment of protein sequences\n"
        "3. Phylogenetic tree inference balancing accuracy and runtime for a large protein dataset\n"
}

class GenomeAssemblyPipeline(BaseModel):
    error_correction: Literal[
        "musket", "bless", "lighter", "hitec", "reptile"
    ]

    assembler: Literal[
        "spades", "abyss", "sga", "edenanonstrict", "cortex"
    ]
    
    def to_list(self):
        return [self.error_correction, self.assembler]

class VariantDetectionPipeline(BaseModel):
    aligner: Literal[
        "bwa", "bwamem", "gsnap", "novoalign"
    ]
    cnv_detection: Literal[
        "freec", "cnvnorm", "exomecnv", "cnvkit", "cnvseq"
    ]
    structural_variant: Literal[
        "lumpy", "delly", "manta", "svaba", "breakdancer"
    ]
    
    def to_list(self):
        return [
            self.aligner,
            self.cnv_detection,
            self.structural_variant
        ]

class RNASeqPipeline(BaseModel):
    aligner: Literal[
        "star", "hisat", "gsnap", "segemehl", "mapsplice"
    ]
    splicing: Literal[
        "mats", "dexseq", "diffsplice", "singe", "dsgseq"
    ]
    fusion: Literal[
        "arriba", "starfusion", "fusioncatcher", "chimpipe", "tophatfusion"
    ]
    expression: Literal[
        "cufflinks", "edger", "deseq2", "rdiffparam", "seqgsea"
    ]

    def to_list(self):
        return [
            self.aligner,
            self.splicing,
            self.fusion,
            self.expression
        ]

class PhylogenyPipeline(BaseModel):
    homology_search: Literal[
        "hmmer", "phmmer", "csblast", "hhsearch", "ncbiblast"
    ]
    msa: Literal[
         "mafft", "muscle", "probcons", "promals", "tcoffee"
    ]
    tree: Literal[
        "raxml", "fasttree", "probtree", "starbeast", "mrpml"
    ]

    def to_list(self):
        return [
            self.homology_search,
            self.msa,
            self.tree
        ]

TASK_TO_MODEL = {
    "genome_assembly": GenomeAssemblyPipeline,
    "variant_detection": VariantDetectionPipeline,
    "RNA-seq": RNASeqPipeline,
    "Phylogeny": PhylogenyPipeline
}

ROLES = {
    "naive": "Suggest a pipeline to solve the task.",
    "expert": "You are an expert bioinformatician. "
    "Suggest a reproducible pipeline based on benchmark performance and suitability "
    "for the specific experimental context described, prioritizing accuracy",
    }



MODELS = [
    "google/gemini-2.5-pro",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet-4-6",
    "deepseek/deepseek-chat"
]

N_RUNS = 20

BASE_DIR = Path(__file__).parent.parent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(ROOT,"data/raw/ai_bioinfo_pipeline.csv")
SCORED_DATA_PATH = os.path.join(ROOT,"data/benchmark_data/ai_bioinfo_pipeline_benchmarks.csv") 

ACC_DATA_PATH = os.path.join(ROOT,"data/benchmarks/rawRankSpeedData2005-2020.tsv")
CIT_DATA_PATH = os.path.join(ROOT,"data/benchmarks/speed-vs-accuracy-toolinfo2005-2020.tsv")


BENCHMARK_PATH = os.path.join(ROOT,"data/benchmarks/tool_benchmark.csv")
BENCHMARK_FILTERED_PATH = os.path.join(ROOT,"data/benchmarks/tool_benchmark_filtered.csv")
BENCHMARK_MANUAL_PATH = os.path.join(ROOT,"data/benchmarks/benchmark_manual.csv")
BENCHMARK_FINAL_PATH = os.path.join(ROOT,"data/benchmarks/tool_benchmark_final.csv")

FIGURES_PATH = os.path.join(ROOT,"outputs/figures_")
FIGURES_SUP_PATH = os.path.join(ROOT,"outputs/figures_sup")


