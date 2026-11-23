library(data.table)
library(purrr)
library(ggplot2)
library(viridis)
library(stringr)
#library(tidyverse)
library(Seurat)


################################################################################  
########################## clean DNA fragments #################################  
################################################################################  

############################### Bulk data ######################################  

in.file.dir <- "~/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/bulk/mtx"  
genomes<- "hs"
file.pattern <-  paste0(genomes,"_sorted_rmdup.reads.split.bc.uniq.tsv.gz")
#output files:
out.file.dir <- dirname(in.file.dir) #"/bi/home/wangy/projects/ctr_seq/08.matrix"
out.file.folder <- "fragments"

if (!file.exists(paste0(out.file.dir,"/",out.file.folder))){
  dir.create(paste0(out.file.dir,"/",out.file.folder))
}

files<-list.files(in.file.dir,pattern = file.pattern,full.names = T)

fread(files[1])

for (f1 in files[3:11]) {
foo<-lapply(f1, function(i) fread(i, fill=T, sep="\t", nThread=8 )%>% 
              setnames(c("chr","start","end","umi","cellid","bc4")) %>%  ## deduplicated: umi_chr_pos
              .[,sample.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,3] ] %>% #4
              .[,lib.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,2] ]%>% #3
              .[,mapping.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,4] ]%>%
              .[,sub.lib:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,1] ] #5
)%>%
  rbindlist()
foo<-unique(foo)
foo[chr=="chrM"]
foo<-foo[chr %in% paste0("chr",c(1:22,"X","Y"))]
foo$umi<-NULL
foo<-unique(foo)
foo[sub.lib %in% c("D0") &bc4==1, ab.type:="H3K4me1"]
foo[sub.lib %in% c("D0") &bc4==2, ab.type:="H3K4me3"]
foo[sub.lib %in% c("D0") &bc4==3, ab.type:="H3K27ac"]
foo[sub.lib %in% c("D0") &bc4==4, ab.type:="H3K27me3"]
foo[sub.lib %in% c("D0") &bc4==5, ab.type:="H3K36me3"]
foo[sub.lib %in% c("D0") &bc4==6, ab.type:="IgG"]

foo<-foo[!is.na(ab.type)]

foo.tmp<-foo[,.(chr,start,end , id=paste0(sub.lib, "_",ab.type) ,num=1)]%>% split(by=c("id"),keep.by = T)

walk2(foo.tmp, paste0(out.file.dir,"/",out.file.folder,"/", names(foo.tmp),".fragments.tsv"), fwrite,sep="\t",col.names=F)
}

############################### single cell data ###############################  

in.file.dir <- "~/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/paired"  
in.file.dir <- paste0(in.file.dir ,"/mtx")  #"/bi/home/wangy/projects/ctr_seq/03.mm10_mapping"
genomes<- "hs"
file.pattern <-  paste0(genomes,"_sorted_rmdup.reads.split.bc.uniq.tsv.gz")
#output files:
out.file.dir <- dirname(in.file.dir) #"/bi/home/wangy/projects/ctr_seq/08.matrix"
out.file.folder <- "fragments"
if (!file.exists(paste0(out.file.dir,"/",out.file.folder))){
  dir.create(paste0(out.file.dir,"/",out.file.folder))
}
files<-list.files(in.file.dir,pattern = file.pattern,full.names = T)
for (fi in files[c(1:8)]){
  #  fi=files[7]
  foo<-lapply(fi, function(i) fread(i, fill=T, sep="\t", nThread=8 )%>% 
                setnames(c("chr","start","end","umi","cellid","bc4")) %>%  ## deduplicated: umi_chr_pos
                .[,sample.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,1] ] %>% #4
                .[,lib.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,2] ]%>% #3
                .[,mapping.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,4] ]%>%
                .[,sub.lib:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,3] ] #5
  )%>%
    rbindlist()
  foo<-foo[chr %in% paste0("chr",c(1:22,"X","Y"))]
  foo$umi<-NULL
  foo<-unique(foo)
  # all reads in cell
  tmp<-foo[,.N,.(cellid,sub.lib,sample.type)]%>% setorder(N)
  tmp[ ,"neworder":=rank(-N)]
  tmp[,"deduplicated_reads":=N]
  tmp[,N:=NULL]
  foo[bc4 %in% c("1"), ab.type:="H3K4me1"]
  foo[bc4 %in% c("2"), ab.type:="H3K4me3"]
  foo[bc4 %in% c("3"), ab.type:="H3K27ac"]
  foo[bc4 %in% c("4"), ab.type:="H3K27me3"]
  foo[bc4 %in% c("5"), ab.type:="H3K36me3"]
  foo[bc4 %in% c("6"), ab.type:="rbIgG"]
  foo<-foo[!is.na(ab.type)]
  #read.summary<-fread("/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/paired/reads.summary/rna.dna.reads.summary.tsv")
  #kept.cell<-read.summary[(pass.DNA&pass.RNA ), .(sample.type.y,new.cellid) ]
  foo[,new.cellid:=paste0(cellid,":",sub.lib) ]
  # rm no ab sample
  tmp<-foo[!is.na(ab.type),.N,.(cellid,sub.lib,sample.type,ab.type)]%>% setorder(N)
  tmp[,new.cellid:=paste0(cellid,":",sub.lib) ]
  tmp[,all.reads.per.cell:=sum(N),new.cellid]
  tmp<-dcast(tmp,new.cellid+all.reads.per.cell~ab.type,value.var = "N")
  cutoff=2500
  tmp[,cutoff:=cutoff]
  fwrite(tmp,paste0(in.file.dir,"/",sample.name,".cell.meta.info.allcell.tsv"),sep="\t" )
  foo<-foo[new.cellid %in% tmp[all.reads.per.cell>=cutoff,new.cellid]]
  foo.tmp<-foo[,.N,.(chr,start,end,new.cellid ,  ab.type )]%>% split(by=c( "ab.type"),keep.by = F)
  walk2(foo.tmp, paste0(out.file.dir,"/",out.file.folder,"/",sample.name,"_", names(foo.tmp),".fragments.tsv"), fwrite,sep="\t",col.names=F)
}

in.file.dir <- "/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/paired/fragments"  
out.file.dir <- (in.file.dir) #"/bi/home/wangy/projects/ctr_seq/08.matrix"
out.file.folder <- "fragments"
if (!file.exists(paste0(out.file.dir,"/",out.file.folder))){
  dir.create(paste0(out.file.dir,"/",out.file.folder))
}
files<-list.files(in.file.dir,pattern = ".fragments.tsv.gz",full.names = T)

read.summary<-fread("/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/paired/reads.summary/rna.dna.kept.summary.tsv")
read.summary<-read.summary[lib.type=="RNA"& deduplicated_reads>=2000]
for (fi in c("H3K27ac","H3K27me3","H3K36me3","H3K4me3","H3K4me1","rbIgG")){
  # fi= "H3K27ac" #H3K27ac|
  fi=files[grep(fi,files)]
  foo<-lapply(fi, function(i) fread(i, fill=T, sep="\t", nThread=8 )%>% 
                setnames(c("chr","start","end","cellid","num")) %>%  ## deduplicated: umi_chr_pos
                .[,sample.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,1] ] %>% #4
                .[,ab.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,3] %>% str_split(.,'[.]',simplify = T)%>% .[,1] ]%>%
                .[,sub.lib:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,2] ] #5
  )%>%
    rbindlist()
  foo<-merge(foo,read.summary[,.(cellid=new.cellid,sample.type)],by=c("cellid"))
  foo.tmp<-foo[,.N,.(chr,start,end,cellid , sample.type.y,  ab.type )]%>% split(by=c("sample.type.y", "ab.type"),keep.by = F)
  walk2(foo.tmp, paste0(out.file.dir,"/",out.file.folder,"/", names(foo.tmp),".fragments.tsv"), fwrite,sep="\t",col.names=F)
}



