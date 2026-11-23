source("~/scripts/utility.R")
#sample E8.5, HNES1
library(data.table)
library(purrr)
library(ggplot2)
library(viridis)
library(stringr)
library(ggpointdensity)

########################################################################
################# prepare matrix and sample information ################
########################################################################

in.file.dir <- "/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/RNA"  


files<-list.files(paste0(in.file.dir ,"/reads.summary"),recursive = T,pattern="xls",full.names = T)#recursive = T,

foo<-lapply(files, function(i) fread(i,select = c(1,2,3,5)) %>%  ## deduplicated: umi_chr_pos
              setnames(c("cellid", "all_reads", "mapped_reads", "deduplicated_reads")) %>%
              .[,c("mapping_rate","duplication_rate"):=list(mapped_reads/all_reads, 1-deduplicated_reads/mapped_reads)] %>% 
              #  .[,sub.lib:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,3] ] %>%
              .[,sample.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,4] ] %>% #4
              .[,lib.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,5] ]%>% #3
              .[,mapping.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,9] ]%>%
              .[,sub.lib:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,6] ] #5
)%>%
  rbindlist()

foo<-foo[deduplicated_reads>0]

foo[lib.type=="R",lib.type:="RNA"]

in.file.dir <- "/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/paired"  

files<-list.files(paste0(in.file.dir ,"/reads.summary"),recursive = T,pattern="xls",full.names = T)#recursive = T,

foo.DNA<-lapply(files, function(i) fread(i,select = c(1,2,3,5)) %>%  ## deduplicated: umi_chr_pos
              setnames(c("cellid", "all_reads", "mapped_reads", "deduplicated_reads")) %>%
              .[,c("mapping_rate","duplication_rate"):=list(mapped_reads/all_reads, 1-deduplicated_reads/mapped_reads)] %>% 
              #  .[,sub.lib:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,3] ] %>%
              .[,sample.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,1] ] %>% #4
              .[,lib.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,2] ]%>% #3
              .[,mapping.type:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,4] ]%>%
              .[,sub.lib:=basename(i) %>% str_split(.,'_',simplify = T)%>% .[,3] ] #5
)%>%
  rbindlist()

foo.DNA[lib.type=="D",lib.type:="DNA"]
foo<-rbind(foo,foo.DNA)
############################################
############################################
# for paired end mapping:
foo[lib.type=="DNA",mapped_reads:=mapped_reads/2]
foo[lib.type=="DNA",deduplicated_reads:=deduplicated_reads/2]
foo[lib.type=="DNA",mapping_rate:=mapped_reads/all_reads]
############################################
############################################

foo[lib.type=="RNA"& substring(cellid,10,11) %in% c("01"),sample.type:="Def.endo.D0"]
foo[lib.type=="RNA"& substring(cellid,10,11) %in% c("02") ,sample.type:="Def.endo.D1"]
foo[lib.type=="RNA"& substring(cellid,10,11) %in% c("03"),sample.type:="Def.endo.D2"]
foo[lib.type=="RNA"& substring(cellid,10,11) %in% c("04"),sample.type:="Def.endo.D3"]
foo[lib.type=="DNA"& substring(cellid,10,11) %in% c("01"),ab.type:="H3K4me1"]
foo[lib.type=="DNA"& substring(cellid,10,11) %in% c("02"),ab.type:="H3K4me3"]
foo[lib.type=="DNA"& substring(cellid,10,11) %in% c("03"),ab.type:="H3K27ac"]
foo[lib.type=="DNA"& substring(cellid,10,11) %in% c("04"),ab.type:="H3K27me3"]
foo[lib.type=="DNA"& substring(cellid,10,11) %in% c("05"),ab.type:="H3K36me3"]
foo[lib.type=="DNA"& substring(cellid,10,11) %in% c("06"),ab.type:="rbIgG"]

foo<-foo[!(lib.type=="RNA"&sample.type=="HIST")]
foo<-foo[!(lib.type=="DNA"&is.na(ab.type))]

foo[,new.cellid:=paste0(substring(cellid,1,8),":",sub.lib)]
#foo<-foo[!sample.type=="H9"]

#remove rna tag mismatched id
foo[lib.type=="RNA",sum(all_reads), substring(cellid,10,11)]
foo[lib.type=="RNA",ab.type:="RNA"]
foo[,.(sample.type,lib.type,mapping.type)] %>% unique()
foo[,ratio:=deduplicated_reads/sum(deduplicated_reads),.(lib.type,sample.type,ab.type,new.cellid)]
foo[ratio<1]
foo[,ratio:=NULL]

foo[,all.reads.per.cell:=sum(deduplicated_reads),.(new.cellid,sample.type,lib.type)]
options(scipen=999) 

fwrite(foo,paste0(in.file.dir,"/reads.summary/clean.all.reads.summary.Def.endo1.tsv"),sep = "\t")

###############################################################################
############use knee plot to find cutoff for cellid vs background##############
###############################################################################

tmp<-foo[, .(all.reads.per.cell),.(sample.type,lib.type,new.cellid,sub.lib)]%>% unique() %>% setorder(all.reads.per.cell)
tmp[,"neworder":=rank(-all.reads.per.cell),.(lib.type,sample.type,sub.lib)]#sample.type,sub.lib,

cut.offs<-c()
tmp.num<-c()
for (cut.offs in c(1000,2000,3000)) {
  tmp.n<-tmp[,.N,.(lib.type,sample.type,sub.lib,all.reads.per.cell>=cut.offs)] %>% .[all.reads.per.cell==T] 
  tmp.n[,cutoff:=cut.offs]
  tmp.num<-rbind(tmp.num,tmp.n)
}
pdf(paste0(in.file.dir,"/reads.summary/cell.vs.background.all.reads.per.cell.Def.endo2.pdf"),width = 9,height=3)
ggplot(tmp,aes(neworder,all.reads.per.cell))+
  geom_line(color="red")+
  #  geom_hline(yintercept = 5000,linetype = "dashed",alpha=0.3)+
  geom_hline(yintercept = 3000,linetype = "dashed",alpha=0.3)+
  geom_hline(yintercept = 2000,linetype = "dashed",alpha=0.3)+
  geom_hline(yintercept = 1000,linetype = "dashed",alpha=0.3)+
   facet_grid(sub.lib~lib.type+sample.type,scales = "free",space = "free",drop = T)+
  
  #facet_wrap(sample.type~lib.type)+
  geom_text(data = tmp.num,aes(label=N,x=1000,y=cutoff),color="black",size=3)+
  labs(y="Unique fragments",x="Number of cellids")+
  scale_y_log10()+
  scale_x_log10()+
  theme_bw() +
  theme(
    #    strip.background = element_blank(),
    strip.text = element_text(size=12),
    axis.text.x = element_text(size=12),
    axis.text.y = element_text(size=12),
    axis.title = element_text(size=14),
    legend.justification = "center",
    legend.title = element_blank(),
    legend.text = element_text(size=12),
    panel.background = element_blank()
  ) 
dev.off()

###############################################################################
#####################  matching RNA id and DNA id #############################
###############################################################################
# RNA kept reads 
kept.rna.reads<-2000 #200
# DNA sum reads 
kept.dna.reads<-2500 #200

test<-foo[lib.type=="DNA",.(new.cellid,deduplicated_reads,sample.type,lib.type,all.reads.per.cell,ab.type)] %>% dcast(.,new.cellid+sample.type+all.reads.per.cell~ab.type,value.var = "deduplicated_reads")
test[all.reads.per.cell>=kept.dna.reads]
foo[lib.type=="RNA",ratio:=deduplicated_reads/sum(deduplicated_reads),new.cellid]
foo[lib.type=="RNA" & ratio>=0.5]
test.rna<-foo[lib.type=="RNA" & ratio>0.5,.(new.cellid,cellid,deduplicated_reads,sample.type,lib.type,all.reads.per.cell)] %>% dcast(.,new.cellid+cellid+sample.type+all.reads.per.cell~lib.type,value.var = "deduplicated_reads")
test<-merge(test,test.rna[,.(new.cellid,cellid,sample.type,RNA)],by=c("new.cellid"))
test[all.reads.per.cell>=kept.dna.reads ,pass.DNA:=T]
test[all.reads.per.cell<kept.dna.reads ,pass.DNA:=F]
test[ RNA >=kept.rna.reads,pass.RNA:=T]
test[ RNA <kept.rna.reads,pass.RNA:=F]

test[pass.DNA&pass.RNA]
test[,unique(sample.type)]

########################################################################################################
#####################  scatter plot/density RNA and DNA per cellid combinations ########################
########################################################################################################

for (samples in test[,unique(sample.type.y)] ){
  samples<-"Def.endo.D0"
  #ab.types<-"all.reads.per.cell"
  for (ab.types in colnames(test)[3:9]){
    png(paste0((in.file.dir),"/reads.summary/DNA.RNA.scaterplot.",samples,".",ab.types,".png"),width = 600,height= 600,res=150)
    select_cols <- c("new.cellid",ab.types,"RNA","pass.DNA","pass.RNA")
    to.plot<-test[sample.type.y == samples, ..select_cols] %>% setnames(c("new.cellid","dna.reads","rna.reads","pass.DNA","pass.RNA"))
    to.plot[, plot.color:= paste0("DNA:",pass.DNA,"&RNA:",pass.RNA)]
    nums.to.label<-to.plot[!is.na(dna.reads),.N,.(pass.DNA,pass.RNA)]
    xlim_len<-max(test$all.reads.per.cell) %>% log10()#max(to.plot[!is.na(dna.reads),dna.reads] %>% log10())
    ylim_len<-max(test$RNA) %>% log10() #max(to.plot[!is.na(rna.reads),rna.reads] %>% log10())
    
    nums.to.label[,x.pos:=ifelse(pass.DNA==TRUE,xlim_len-0.5,0.5 )]
    nums.to.label[,y.pos:=ifelse(pass.RNA==TRUE,ylim_len-0.5,0.5 )]
    
    p1<-ggplot(to.plot,aes(log10(dna.reads),log10(rna.reads)))+
      geom_point(aes(color=plot.color),size=0.1)+
      xlim(0, xlim_len)+
      ylim(0, ylim_len)+
      scale_color_manual(values = c("grey90","grey60","grey60","red"))+
      labs(x="log10(Number of DNA reads)",y="log10(Number of RNA reads)",title = paste0(samples, ": ", ab.types," vs RNA"),subtitle = paste0("cutoff: DNA per cell>=",kept.dna.reads," RNA>=",kept.rna.reads))+
      geom_hline(yintercept = log10(kept.rna.reads),linetype="dashed",color="grey")+
      geom_vline(xintercept = log10(kept.dna.reads),linetype="dashed",color="grey")+
      geom_text(data = nums.to.label,aes(label=N,x=x.pos,y=y.pos),color="black",size=6)+
      coord_flip()+
      theme_bw()+
      theme(
        strip.text = element_text(size=12),
        axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=12),
        axis.title = element_text(size=14),
        legend.position = "none",
        panel.grid = element_blank(),
        panel.background = element_blank()
      ) 
    print(p1)     
    
    dev.off()
    
    pdf(paste0((in.file.dir),"/reads.summary/DNA.RNA.scaterplot.density.",samples,".",ab.types,".pdf"),width = 6,height= 6)
    p2<-ggplot(to.plot,mapping = aes(y=log10(dna.reads),x=log10(rna.reads)))+
      xlim(0, ylim_len)+
      ylim(0, xlim_len)+
      geom_pointdensity()+
      scale_color_viridis()+
      labs(y="log10(Number of DNA reads)",x="log10(Number of RNA reads)",title = paste0(samples, ": ", ab.types," vs RNA"),subtitle = paste0("cutoff: DNA per cell>=",kept.dna.reads," RNA>=",kept.rna.reads))+
        coord_fixed(ratio =xlim_len/ ylim_len)+
      theme_bw()+
      theme(
        strip.text = element_text(size=12),
        axis.text.x = element_text(size=12),
        axis.text.y = element_text(size=12),
        axis.title = element_text(size=14),
        panel.grid = element_blank(),
        panel.background = element_blank()
      ) 
    print(p2)
    dev.off()
    
  }
}

fwrite(test,paste0((in.file.dir),"/reads.summary/rna.dna.reads.summary.tsv"),sep = "\t")
keptcell<-test[pass.DNA&pass.RNA, new.cellid]
to.plot<-foo[new.cellid %in%  keptcell] 
keptrnacell<-test[pass.DNA&pass.RNA, paste0(new.cellid,"_",cellid)]
to.plot.rna<-to.plot[ab.type=="RNA" & paste0(new.cellid,"_",cellid) %in% keptrnacell]
to.plot.dna<-merge(to.plot[lib.type=="DNA", -("sample.type")],to.plot.rna[,.(new.cellid,sample.type)],by="new.cellid")
to.plot<-rbind(to.plot.rna,to.plot.dna)
fwrite(to.plot,paste0((in.file.dir),"/reads.summary/rna.dna.kept.summary.tsv"),sep = "\t")


############################################################################################################
############################## plot summary of kept cell ###################################################
############################################################################################################

to.plot<-to.plot[, c("sub.lib","cellid","all.reads.per.cell"):=list(NULL,NULL,NULL)] %>% melt() 
to.plot[,.(new.cellid,sample.type)]%>% unique() %>% .[,.N,sample.type]
to.plot[,unique(variable)]
to.plot[variable %in% c("all_reads","mapped_reads","deduplicated_reads"),unit:="log10"]
to.plot[!variable %in% c("all_reads","mapped_reads","deduplicated_reads"),value:=(value)*100]
to.plot[!variable %in% c("all_reads","mapped_reads","deduplicated_reads"),unit:="%"]
#to.plot[lib.type=="RNA",ab.type:="RNA"]
to.plot[,unique(ab.type)]
to.plot$ab.type<-factor(to.plot$ab.type,levels =
                          c("RNA","rbIgG","H3K4me1","H3K4me3","H3K27ac","H3K27me3","H3K36me3")     
)
means <- aggregate(value ~sample.type+variable+unit+ ab.type, to.plot, mean) %>% as.data.table() %>% .[,value:=round(value*100)/100]
to.plot[variable %in% c("all_reads","mapped_reads","deduplicated_reads"),value:=log10(value)]
means[unit=="log10",value:=round(value)]

pdf(paste0(in.file.dir,"/reads.summary/boxplot.mapping.states.pdf"),width = 12,height= 10)
for (samples in to.plot[,unique(sample.type)]){
  p1<-ggplot(to.plot[sample.type==samples],aes(ab.type,value))+
    #  geom_jitter(aes(color=ab.type),alpha=0.6,size=0.1)+
    geom_violin(aes(color=ab.type))+
    #    scale_color_viridis(discrete = TRUE, begin = 0.1, end = 0.9, option = "magma", alpha = 0.8) +
    geom_boxplot(color="grey",alpha=0.5,fill=NA,outlier.colour = NA,width=0.2)+
    #   stat_summary(fun.y=mean, geom="point", shape=19, size=2,alpha=0.8)+
    geom_text(data = means[sample.type==samples], aes(label = value, y = 3 *0.8))+
    #  geom_text(data = medians[sample.type==samples], aes(label = value, y = value *0.4))+
    labs(x="",title = paste0(samples))+#,subtitle = paste0("cutoff: DNA per cell>=",kept.dna.reads," RNA>=",kept.rna.reads))+
    facet_grid(variable+unit~.,scale="free_y")+
    theme_bw() +
    theme(
      #    strip.background = element_blank(),
      strip.text = element_text(size=12),
      axis.text.x = element_text(size=12, angle = 45, vjust = 0.5),
      axis.text.y = element_text(size=12),
      axis.title = element_text(size=14),
      legend.position = "none",
      panel.background = element_blank()
    ) 
  print(p1)
}
dev.off()


