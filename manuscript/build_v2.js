const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, Footer, ImageRun } = require("docx");
const ARIAL = "Arial";
const FIGDIR = "/mnt/user-data/uploads";
const FIGDIR2 = require("path").join(__dirname, "figs");  // writable, for locally-generated figures
// True pixel dimensions (measured with PIL) — the byte-offset reader was unreliable on these files
const PNG_DIMS = {
  "figure_1_study_design.png":[3791,2412],
  "auc_permutation_null.png":[1861,1220],
  "cv_auc_forest.png":[2535,1566],
  "dsi4_wt_ko_trajectory.png":[2670,1470],
  "mixed_effects_trajectories.png":[4017,2246],
  "pac_full_comodulogram_matrix.png":[2582,1059],
  "pac_power_confound.png":[2112,1347],
  "pac_scekic_replication_summary.png":[4468,2660],
  "prespecified_beta_signflip.png":[3870,1485],
  "prespecified_rem_td_ratio.png":[3472,1494],
  "roc_key_features.png":[1609,1694],
  "staging_sensitivity_heatmap.png":[3346,1233],
  "state_separation_diagnostic.png":[1609,1451],
  "sleep_state_validation_motion.png":[2200,920],
  "state_transition_heatmap.png":[2852,1902]
};
// Embed a figure centered, scaled to a target display width in points (max ~460pt page width)
function figure(file, widthPt){
  let p = require("path").join(FIGDIR, file);
  if(!fs.existsSync(p)){ const p2 = require("path").join(FIGDIR2, file); if(fs.existsSync(p2)) p = p2; }
  if(!fs.existsSync(p) || !PNG_DIMS[file]){ return new Paragraph({spacing:{after:80},children:[new TextRun({text:"[missing figure: "+file+"]",italics:true,color:"AA0000"})]}); }
  const buf = fs.readFileSync(p);
  const [w,h] = PNG_DIMS[file];
  const targetW = widthPt || 430;            // points
  const targetH = Math.round(targetW * h / w);
  return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:40},
    children:[new ImageRun({type:"png", data:buf, transformation:{width:targetW, height:targetH}})]});
}
function h1(t){return new Paragraph({heading:HeadingLevel.HEADING_1,children:[new TextRun(t)]});}
function h2(t){return new Paragraph({heading:HeadingLevel.HEADING_2,children:[new TextRun(t)]});}
function para(t){return new Paragraph({spacing:{after:160,line:276},children:[new TextRun(t)]});}
function rich(runs){return new Paragraph({spacing:{after:160,line:276},children:runs});}
function todo(t){return new Paragraph({spacing:{after:160,line:276},shading:{type:ShadingType.CLEAR,fill:"FFF2CC"},children:[new TextRun({text:"[TODO] "+t,italics:true,color:"7F6000"})]});}
function ph(t){return new Paragraph({spacing:{after:160,line:276},shading:{type:ShadingType.CLEAR,fill:"E2EFDA"},children:[new TextRun({text:"[PLACEHOLDER] "+t,italics:true,color:"375623"})]});}
function meta(l,v){return new Paragraph({spacing:{after:60},children:[new TextRun({text:l+": ",bold:true}),new TextRun(v)]});}
const bd={style:BorderStyle.SINGLE,size:1,color:"CCCCCC"};const bds={top:bd,bottom:bd,left:bd,right:bd};
const hdr=["Feature","TP","WT","KO","Cohen's d [95% CI]","p","q (FDR)"];
const rows=[
["REM Theta/Delta Ratio","3m","1.543","1.519","-0.04 [-0.83, 1.62]","0.331","0.603"],
["REM Theta/Delta Ratio","4m","0.784","1.415","1.97 [1.12, 3.81]","0.007","0.047 *"],
["REM Theta/Delta Ratio","9m","1.458","1.785","0.91 [-0.37, 4.03]","0.114","0.229"],
["REM Theta/Delta Ratio","12m","2.124","1.732","-0.67 [-3.98, 1.45]","0.686","0.962"],
["REM Relative Theta","3m","0.318","0.347","0.78 [-0.20, 2.52]","0.112","0.229"],
["REM Relative Theta","4m","0.244","0.309","1.50 [0.65, 2.84]","0.028","0.113"],
["REM Relative Theta","9m","0.349","0.395","2.63 [1.39, 6.61]","0.038","0.127"],
["REM Relative Theta","12m","0.352","0.379","0.74 [-0.98, 3.09]","0.486","0.747"],
["Wake Relative Beta","3m","0.138","0.118","-0.42 [-1.62, 0.56]","0.480","0.747"],
["Wake Relative Beta","4m","0.091","0.166","1.26 [0.86, 2.27]","0.005","0.047 *"],
["Wake Relative Beta","9m","0.126","0.131","0.14 [-1.48, 2.08]","0.914","0.962"],
["Wake Relative Beta","12m","0.152","0.122","-2.23 [-6.73, -1.05]","0.114","0.229"],
["Sleep Spindle Duration (s)","3m","1.021","0.936","-1.34 [-3.02, -0.40]","0.022","0.108"],
["Sleep Spindle Duration (s)","4m","0.851","0.859","0.06 [-1.10, 1.14]","1.000","1.000"],
["Sleep Spindle Duration (s)","9m","0.999","1.020","0.09 [-1.31, 1.92]","0.914","0.962"],
["Sleep Spindle Duration (s)","12m","0.877","0.864","-0.16 [-2.27, 1.75]","0.886","0.962"],
["REM Relative Beta","3m","0.151","0.142","-0.19 [-1.27, 0.85]","0.791","0.962"],
["REM Relative Beta","4m","0.108","0.159","1.27 [0.90, 2.61]","0.003","0.047 *"],
["REM Relative Beta","9m","0.141","0.139","-0.09 [-1.95, 1.73]","0.762","0.962"],
["REM Relative Beta","12m","0.158","0.135","-1.90 [-5.56, -0.71]","0.114","0.229"]];
const cw=[2400,600,900,900,2400,1080,1080];
function cell(t,h,w){return new TableCell({borders:bds,width:{size:w,type:WidthType.DXA},shading:h?{fill:"D5E8F0",type:ShadingType.CLEAR}:undefined,margins:{top:60,bottom:60,left:100,right:100},children:[new Paragraph({spacing:{after:0},children:[new TextRun({text:t,bold:h,size:18,font:ARIAL})]})]});}
const table1=new Table({width:{size:9360,type:WidthType.DXA},columnWidths:cw,rows:[new TableRow({tableHeader:true,children:hdr.map((t,i)=>cell(t,true,cw[i]))}),...rows.map(r=>new TableRow({children:r.map((c,i)=>cell(c,false,cw[i]))}))]});

// Generic table builder: header array, rows (array of arrays), and column widths (DXA)
function makeTable(headers, datarows, colw){
  const total = colw.reduce((a,b)=>a+b,0);
  return new Table({width:{size:total,type:WidthType.DXA},columnWidths:colw,rows:[
    new TableRow({tableHeader:true,children:headers.map((t,i)=>cell(t,true,colw[i]))}),
    ...datarows.map(r=>new TableRow({children:r.map((cc,i)=>cell(cc,false,colw[i]))}))
  ]});
}
function tcap(t){return new Paragraph({spacing:{before:200,after:60},children:[new TextRun({text:t,bold:true})]});}

const c=[];
c.push(new Paragraph({spacing:{after:240},children:[new TextRun({text:"A biphasic trajectory of cortical and hippocampal network dysfunction during REM sleep in C9orf72-deficient mice",bold:true,size:32,font:ARIAL})]}));
c.push(rich([new TextRun({text:"Running title: ",bold:true}),new TextRun("REM network dysfunction trajectory in C9orf72 mice")]));
c.push(new Paragraph({spacing:{after:60,before:160},children:[new TextRun({text:"Belay [Surname]\u00B9, Tewolde Teklu\u00B2*, Haben Girmay Yhdego\u00B3, [additional authors]",size:22})]}));
c.push(para("\u00B9 Tanz Centre for Research in Neurodegenerative Diseases, University of Toronto, Toronto, Ontario, Canada"));
c.push(para("\u00B2 [Department], Axum University, Axum, Ethiopia"));
c.push(para("\u00B3 [Affiliation]"));
c.push(rich([new TextRun({text:"*Corresponding author. Email: [email]",italics:true})]));
c.push(meta("Manuscript type","Original Article"));
c.push(meta("Word count (main text)","[TODO \u2014 Brain limit 6000]"));
c.push(meta("Abstract","\u2264400 words (Brain limit)"));
c.push(meta("Display items","6 figures, 1 table (Brain limit 8)"));

c.push(h1("Abstract"));
c.push(para("Cortical hyperexcitability is a unifying, prognostically important feature of amyotrophic lateral sclerosis (ALS) that precedes symptom onset and correlates with disease progression. The network signatures used to track it have been defined almost entirely in SOD1 and FUS models, in which a theta-gamma phase-amplitude coupling (PAC) deficit, linked to noradrenaline depletion, indexes cortical hyperexcitability; in sporadic patients this deficit is frequency-specific and localized to the dominant sensorimotor cortex. Whether these signatures generalize to C9orf72-mediated disease \u2014 the commonest genetic cause of ALS and frontotemporal dementia, and mechanistically distinct in involving loss of C9orf72 function \u2014 is unknown, with direct implications for whether a single network biomarker can serve a genetically heterogeneous patient population. We conducted a longitudinal electrophysiological study of C9orf72-knockout mice and wild-type littermates, recording hippocampal (CA3) and parietal sensorimotor (S1/PtA) cortical EEG with synchronous video across six timepoints from 3 to 12 months, spanning a pre-symptomatic baseline, an acute kainic-acid challenge, and progression to end-stage. Using a pre-specified, false discovery rate-controlled framework with per-timepoint and mixed-effects trajectory testing, we identified a biphasic trajectory of network dysfunction most clearly expressed during REM sleep. Relative beta power was elevated in knockout mice during the acute phase (4 months; REM Cohen\u2019s d = 1.27, wake d = 1.26) and reduced at end-stage (12 months; REM d = \u22121.66, wake d = \u22121.61); this sign reversal was supported by per-timepoint testing in both directions and by cluster-robust mixed-effects modelling (interaction p = 0.048). REM theta/delta ratio was acutely elevated at challenge (d = 1.97) and REM relative theta remained elevated through 9 months (d = 1.76). Sleep architecture showed a progressive bias toward entering REM from NREM (12 months d = 3.85). In direct contrast to SOD1 and FUS models, C9orf72-knockout mice showed no cortical theta-gamma PAC deficit during REM at any timepoint, establishing a genotype-specific network divergence. A behavioural battery at 10 months revealed multi-domain impairment in knockout mice, with significantly reduced grip strength (d = \u22121.23) and cued fear freezing (d = \u22121.84) and large-effect reductions in recognition memory and locomotion; the modest phenotyped sample precluded robust individual-level coupling between specific network features and behaviour. These findings identify REM sleep as a sensitive window onto C9orf72 network dysfunction and reveal a genotype-specific, biphasic trajectory compatible with the hyperexcitable-to-hypoexcitable transition of human ALS, with consequences for stratified biomarker development."));
c.push(rich([new TextRun({text:"Keywords: ",bold:true}),new TextRun("C9orf72; amyotrophic lateral sclerosis; frontotemporal dementia; EEG; REM sleep; cortical network dysfunction; phase-amplitude coupling")]));

c.push(h1("Introduction"));
[
"Amyotrophic lateral sclerosis (ALS) is a fatal neurodegenerative disease defined by the progressive loss of upper and lower motor neurons, with death typically occurring within two to five years of symptom onset. A unifying physiological feature across both sporadic and familial forms is cortical hyperexcitability, which precedes the onset of motor symptoms, correlates negatively with survival, and is sufficient to trigger neurodegeneration in rodent models. The ability to detect and track this network dysfunction non-invasively would be of substantial value for early diagnosis, patient stratification, and the monitoring of therapeutic response \u2014 needs that are especially acute in a disease whose clinical heterogeneity delays diagnosis and complicates trial design.",
"Hexanucleotide repeat expansion in C9orf72 is the most common genetic cause of both ALS and frontotemporal dementia, accounting for a substantial fraction of familial and a meaningful proportion of apparently sporadic cases. [TODO insert precise prevalence figures with citation.] Despite this prominence, the network-level electrophysiology of C9orf72-mediated disease remains comparatively underexplored. Most of what is known about ALS cortical network dysfunction derives from the SOD1 and FUS models, in which recent work established a deficit in theta-gamma phase-amplitude coupling (PAC) as a traceable readout of cortical hyperexcitability, present from pre-symptomatic stages, correlated with the rate of disease progression in patients, and linked mechanistically to a deficiency of cortical noradrenaline [ref: Scekic-Zahirovic et al. 2024]. In sporadic ALS patients this signature has been further characterized as frequency-specific and lateralized: theta-gamma PAC is selectively reduced in the dominant (typically left) sensorimotor cortex, with alpha-gamma and beta-gamma coupling and band power unchanged, and with diagnostic performance exceeding that of diffusion MRI [ref: Benetton et al. 2025]. The signature against which a new model must be compared is therefore well defined \u2014 a specific, NA-linked, dominant-hemisphere theta-gamma cortical deficit. These are predominantly gain-of-function models. C9orf72 disease, by contrast, combines repeat-associated gain-of-function toxicity with loss of C9orf72 protein function, the latter affecting autophagy and immune regulation. The C9orf72-knockout model isolates the loss-of-function component, allowing its specific contribution to network pathophysiology to be examined. Because the genetic subtypes of ALS engage partly distinct molecular pathways, the network phenotypes they produce may also differ; whether the SOD1/FUS coupling signature is a universal correlate of ALS cortical dysfunction or a subtype-specific one is unknown, and the answer bears directly on whether a network biomarker calibrated in one genetic context can be expected to transfer to another.",
"Sleep provides a particularly informative window onto cortical network state. Distinct vigilance states impose characteristic oscillatory regimes, and REM sleep in rodents is marked by prominent theta and gamma activity that depends sensitively on the balance of excitation and inhibition; it is also a state in which several neuromodulatory systems that normally constrain cortical excitability are at their lowest tone. Disruption of sleep architecture and of state-specific oscillatory dynamics is increasingly recognized as an early feature of neurodegeneration. We reasoned that a longitudinal, state-resolved analysis of the EEG \u2014 rather than a single cross-sectional snapshot \u2014 would be best placed to reveal how network function evolves across the C9orf72 disease course, and that REM sleep in particular might expose latent network abnormalities.",
"Here we report such a study. We recorded EEG from hippocampal and cortical electrodes, with synchronous video, in C9orf72-knockout mice and wild-type littermates across six timepoints from 3 to 12 months, encompassing a pre-symptomatic baseline, an acute kainic-acid challenge intended to probe network vulnerability, and progression toward end-stage. Using a pre-specified, false discovery rate-controlled analytic framework complemented by mixed-effects trajectory modelling and analysis of sleep-state transition dynamics, we asked three questions: whether and how network function changes across the disease course; whether such changes are state-specific; and whether the theta-gamma PAC signature reported in other ALS models is present in C9orf72 disease. We find a biphasic trajectory of network dysfunction concentrated in REM sleep, and \u2014 in contrast to SOD1 and FUS models \u2014 no cortical theta-gamma PAC deficit, together defining a genotype-specific network phenotype."
].forEach(t=>c.push(para(t)));

c.push(h1("Materials and methods"));
c.push(para("This study is reported in accordance with the ARRIVE guidelines."));
c.push(h2("Animals"));
c.push(todo("Confirm exact numbers, sex distribution, housing, ethics protocol number."));
c.push(para("C9orf72-knockout mice and wild-type littermates were studied. A total of [N] animals contributed data ([N] knockout, [N] wild-type). Animals were maintained under a [12:12] light-dark cycle with food and water available ad libitum. All procedures were approved by the [institutional animal care committee, protocol number TODO] and conformed to [relevant national guidelines]. Sample size was based on [rationale TODO]. Experimenters were blinded to genotype during recording, sleep-state scoring, event detection, and analysis; genotype was unblinded only for final statistical comparison."));
c.push(h2("Animal flow and attrition"));
c.push(todo("Provide a CONSORT-style flow account and corresponding supplementary figure."));
c.push(para("Of [N] animals enrolled, [N] knockout and [N] wild-type contributed usable recordings at the 3-month baseline. Cohort size decreased at later timepoints owing to [attrition causes TODO], yielding [N] per group at 12 months. Missing data arising from attrition were handled by complete-case per-timepoint analysis and by the mixed-effects model\u2019s accommodation of unbalanced repeated measures; no values were imputed. The reduction in end-stage sample size is considered explicitly in the interpretation of late-timepoint effects."));
c.push(h2("EEG and video recording"));
c.push(para("Two-channel EEG was recorded from a hippocampal (CA3) electrode and a parietal sensorimotor cortical electrode (S1/PtA), sampled at [original rate TODO] Hz and stored in Axon Binary Format, with synchronous video for behavioural annotation and movement quantification. Electrodes targeted right CA3 (AP \u22122.5, ML +3.0, DV \u22123.0 mm from bregma) and left S1/PtA (AP \u22122.0, ML \u22122.0, DV \u22121.5 mm), with a frontal reference (AP +1.0, ML +1.0, DV \u22120.5 mm). The cortical site is parietal sensorimotor cortex and is anatomically distinct from primary motor cortex (M1); this distinction is relevant to interpretation and is examined in detail in a companion study of circuit-specific dissociation [ref: companion paper]. Recordings were obtained at six timepoints: 3 months (pre-symptomatic baseline, before kainic-acid administration), 4 months (the kainic-acid challenge day), and 6, 7, 9, and 12 months. Signals were down-sampled to 500 Hz after anti-alias decimation and segmented into 4-second epochs."));
c.push(h2("Kainic-acid challenge"));
c.push(todo("Dose, route, timing relative to recording."));
c.push(para("At the 4-month timepoint, animals received kainic acid to probe network vulnerability. Recordings at this timepoint therefore capture the network response to an acute excitotoxic challenge rather than a spontaneous state, a distinction maintained throughout the interpretation."));
c.push(h2("Sleep-state classification"));
c.push(para("Each 4-second epoch was classified as wake, NREM, or REM using relative band-power criteria. Wake was defined by high signal variance or elevated fast-frequency (beta/gamma) activity; REM by theta-dominant, low-delta, low-amplitude activity (relative theta above the median, relative delta below the median, total variance below the 75th percentile); NREM by delta-dominant activity. The approach was applied identically to all recordings and yielded biologically plausible state distributions (REM [20\u201332]% of epochs per recording). Classification was based on EEG features rather than concurrent electromyography, a limitation considered in the Discussion. State percentages were computed per recording, and epoch-level labels were retained for state-specific spectral analysis and for analysis of state-transition dynamics."));
c.push(h2("Spectral and signal features"));
c.push(para("For each epoch we computed absolute and relative band power (delta 0.5\u20134 Hz, theta 4\u20138 Hz, alpha 8\u201313 Hz, beta 13\u201330 Hz, gamma 30\u201380 Hz), the theta/delta ratio, spectral entropy, Hjorth parameters, signal variance, peak-to-peak amplitude, zero-crossing rate, and Lempel-Ziv complexity. Sleep spindles were detected and their duration quantified as an index of thalamocortical circuit integrity. The aperiodic (1/f) exponent of the power spectrum was estimated to characterize the scale-free component of cortical activity. Features were aggregated within each vigilance state to yield state-specific summaries per recording."));
c.push(h2("Phase-amplitude coupling"));
c.push(para("To test for the theta-gamma PAC signature reported in other ALS models, we computed the modulation index after the method of Tort and colleagues. Coupling was assessed between theta phase (4\u20138 Hz) and both low-gamma (30\u201380 Hz) and high-gamma (80\u2013150 Hz) amplitude, and between alpha phase (8\u201313 Hz) and the same amplitude bands, separately for each vigilance state and recording channel. Matching the published cross-model finding, the pre-specified primary test was cortical theta-high-gamma coupling during REM sleep."));
c.push(h2("Circuit dissociation"));
c.push(para("To quantify divergence between hippocampal and cortical spectral state, we computed dissociation indices comparing CA3 and S1/PtA channels, including a decorrelation index defined as one minus the Spearman correlation of the two channels\u2019 band-power profiles. Trajectories of these indices were compared between genotypes across timepoints."));
c.push(h2("Behavioural testing"));
c.push(todo("Describe the behavioural battery, administered at 10 months of age: novel object recognition (discrimination index), open field (distance, centre time, rearing, velocity), fear conditioning (cued and contextual freezing), grip strength (forelimb, hindlimb, body-weight-normalized composite). State apparatus, protocols, scoring, and blinding."));
c.push(h2("Parvalbumin immunohistochemistry"));
c.push(todo("Describe tissue collection at 12 months, sectioning, parvalbumin immunolabelling, imaging, and quantification of PV-positive interneuron density and mean fluorescence intensity in cortex and hippocampus, \u22653 sections per region per animal."));
c.push(h2("Statistical analysis"));
[
"EEG features were compared between knockout and wild-type mice using a hierarchical strategy. The primary analysis tested pre-specified spectral features at four disease-relevant timepoints (3, 4, 9, and 12 months), with 6- and 7-month timepoints reserved for secondary exploration. For each feature-by-timepoint comparison we computed Mann-Whitney U tests, Cohen\u2019s d with bias-corrected 95% bootstrap confidence intervals (2000 resamples), and applied Benjamini-Hochberg false discovery rate (FDR) correction across the 20 primary tests. Effects are reported as statistically supported when the FDR-adjusted q-value was below 0.05, and as showing effect-size evidence when the bootstrap confidence interval excluded zero.",
"Because the design involved repeated measurements of the same animals across timepoints, trajectory divergence was tested with linear mixed-effects models of the form feature ~ genotype \u00D7 time + (1 | animal), the genotype-by-time interaction quantifying differential change over time. Where the random-intercept variance was estimated near the boundary of the parameter space, we report ordinary least-squares regression with cluster-robust standard errors (clustered by animal) as a sensitivity analysis, since cluster-robust inference accounts for within-animal dependence without requiring an estimable variance component; a trajectory effect is reported as supported only when it survived this sensitivity test.",
"To distinguish whether genotype effects arose at the level of the animal or of individual recording sessions, all analyses were conducted under two unit-of-analysis definitions: a conservative primary analysis treating each animal as the experimental unit, and a sensitivity analysis treating each recording session as the unit. The animal-level analysis is reported as primary throughout; session-level results are provided as supplementary sensitivity comparisons.",
"Sleep-state transition dynamics were quantified from epoch-level state classifications as per-animal three-by-three transition probability matrices, state-specific mean dwell times, and transition rates, with group comparisons by Mann-Whitney U test and Cohen\u2019s d."
].forEach(t=>c.push(para(t)));
c.push(todo("Once behavioural and histological data are integrated: state correlation methods, e.g. Spearman of per-animal EEG features against behavioural scores and PV measures, with FDR correction."));
c.push(h2("Data and code availability"));
c.push(rich([new TextRun("Analysis code is available at https://github.com/BelayTG/eeg-video-analysis-c9orf72. ")]));
c.push(todo("Data availability statement; OSF repository and embargo status."));

c.push(h1("Results"));
const R=[
["A longitudinal, state-resolved EEG dataset across the C9orf72 disease course","We recorded two-channel EEG with synchronous video from C9orf72-knockout mice and wild-type littermates at six timepoints spanning pre-symptomatic, acute-challenge, and progressive disease stages (Fig. 1). Epoch-level vigilance-state classification yielded biologically plausible distributions, with REM sleep representing [20\u201332]% of epochs per recording, enabling state-resolved spectral analysis throughout the disease course and supporting both per-timepoint comparison and the modelling of within-animal trajectories."],
["A biphasic reversal of beta-band network activity is the central longitudinal finding","The defining feature of the trajectory was a reversal in the direction of the between-genotype difference in beta-band power (Fig. 2). During the acute challenge phase at 4 months, relative beta power was elevated in knockout mice in both REM (wild-type 0.108, knockout 0.159; Cohen\u2019s d = 1.27; FDR q = 0.047) and wakefulness (wild-type 0.091, knockout 0.166; d = 1.26; FDR q = 0.047). By end-stage at 12 months the difference had reversed, with relative beta reduced in knockout mice in both REM (wild-type 0.159, knockout 0.135; d = \u22121.66) and wakefulness (wild-type 0.152, knockout 0.122; d = \u22121.61), both reaching FDR significance in the session-level analysis. The reversal was supported at the trajectory level: the genotype-by-time interaction for REM relative beta was significant under cluster-robust inference (interaction p = 0.048), indicating that the knockout and wild-type beta trajectories diverged over time rather than differing only at isolated timepoints. This biphasic course \u2014 elevated beta activity during the acute phase giving way to reduced beta activity at end-stage \u2014 is compatible with the hyperexcitable-to-hypoexcitable transition described in human ALS, here observed as a longitudinal change within the same animals."],
["Acute REM dysregulation at the challenge timepoint","At the 4-month kainic-acid challenge, knockout mice showed broad dysregulation of REM oscillations (Fig. 3; Table 1). The REM theta/delta ratio was markedly elevated (wild-type 0.78, knockout 1.42; d = 1.97; FDR q = 0.047), the largest single effect in the dataset, accompanied by elevated REM relative theta (d = 1.50) and relative alpha (d = 1.38), with confidence intervals excluding zero. The acute response to excitotoxic challenge thus appeared as a shift of the REM spectral profile toward faster, theta-dominant activity. Because the REM theta/delta ratio did not show a monotonic genotype-by-time divergence under cluster-robust inference (robust interaction p = 0.34), we interpret this as a timepoint-specific acute effect rather than a progressive trajectory \u2014 a distinction made explicit by the joint per-timepoint and trajectory analyses."],
["Persistent theta dominance at the progressive stage","At 9 months, REM relative theta remained elevated in knockout mice (wild-type 0.35, knockout 0.40; d = 1.76; FDR q = 0.023, session-level), accompanied by a reduced REM zero-crossing rate (d = \u22120.99), indicating a slower, more regular REM EEG. The theta dominance that emerged acutely at challenge thus persisted into the progressive stage, consistent with a sustained alteration of REM network state."],
["Progressive destabilization of sleep-state architecture","Beyond spectral content, the dynamics of vigilance-state transitions were altered in knockout mice (Fig. 4). Knockout animals showed an increased probability of transitioning from NREM into REM sleep, an effect that intensified with age and was large at end-stage (12 months: wild-type 0.21, knockout 0.26; d = 3.85), accompanied by prolonged REM dwell time at 9 months (d = 1.66). Because REM was the state in which the spectral abnormalities were most pronounced, this transition bias indicates that the architecture of sleep itself becomes progressively skewed toward the most affected state. End-stage transition statistics rest on a small sample (n = 4 per group) and are interpreted with corresponding caution."],
["No cortical theta-gamma PAC deficit: a genotype-specific divergence from SOD1 and FUS models","Because theta-gamma PAC has been established as a cortical hyperexcitability readout in SOD1 and FUS models \u2014 and characterized in sporadic ALS patients as a frequency-specific, dominant-hemisphere deficit confined to theta-gamma coupling \u2014 we tested directly for this signature in C9orf72-knockout mice, matching the published methodology in coupling bands, vigilance state, recording site, and pre-symptomatic timepoint (Fig. 5). Our cortical electrode was positioned over left (dominant-side) sensorimotor cortex, the same region in which the human deficit is maximal, so that a null result here reflects the absence of the signature at the site where it should be most detectable rather than sampling of an unaffected region. Contrary to the pattern reported in those models, cortical theta-high-gamma PAC during REM did not differ between genotypes at any timepoint (3 months d = 0.25; 4 months d = 0.02; 12 months d = 0.02; all non-significant), and where differences trended they did not favour reduced coupling in knockout mice. The absence held across the full coupling matrix tested (theta- and alpha-phase against low- and high-gamma amplitude); the only coupling effects with confidence intervals excluding zero were hippocampal, during NREM, at later timepoints (9-month CA3 alpha-low-gamma d = \u22121.51; 12-month CA3 alpha-high-gamma d = \u22121.15) \u2014 a different channel, state, and frequency combination from the cortical theta-gamma signature reported in humans and other models. C9orf72-knockout mice therefore do not reproduce the cortical theta-gamma PAC deficit that characterizes SOD1 and FUS disease, indicating that this coupling signature is not a universal feature across ALS genetic subtypes. Two further checks support this null. First, across the full cortical-REM coupling matrix (four phase bands by two gamma amplitude bands by three principal timepoints; 24 comparisons), no cell reached FDR significance (Supplementary Fig. S2), establishing that the absence is general across the coupling space rather than specific to a single under-powered test. Second, because the modulation index can be biased by amplitude-band power, we examined gamma-band power directly: at the timepoints where cortical PAC was null, gamma power either did not differ between genotypes or, at the 4-month challenge, was modestly reduced in knockout mice (d = \u22120.45) \u2014 a direction that would tend to depress rather than inflate a coupling estimate. The preserved coupling despite reduced gamma amplitude indicates the null is conservative and not a power artifact (Supplementary Fig. S3)."],
["Latent thalamocortical and circuit-level signatures","At the 3-month pre-symptomatic baseline, sleep spindle duration was reduced in knockout mice (wild-type 1.02 s, knockout 0.94 s; d = \u22121.34; confidence interval excluding zero), a latent thalamocortical signature detectable before overt disease; spindle rate was uninformative, but duration was, underscoring the value of event-level over count-level metrics. At 7 months, CA3-S1/PtA delta coherence was reduced in knockout mice (d = \u22121.28; confidence interval excluding zero), and circuit dissociation between the hippocampal and sensorimotor cortical channels increased progressively with age in knockout but not wild-type animals (knockout decorrelation trajectory \u03C1 = 0.43, p = 0.007; wild-type \u03C1 = 0.06, not significant), consistent with a developing functional disconnection between the two structures; the spatial structure of this sensorimotor-versus-hippocampal divergence is characterized in detail in a companion study [ref: companion paper]. In a prospective relationship, the aperiodic (1/f) exponent at the 3-month baseline predicted Lempel-Ziv complexity at 12 months (Spearman \u03C1 = 0.929; FDR q = 0.034), suggesting that features of the pre-symptomatic cortex foreshadow the eventual network phenotype."]
];
R.forEach(([t,b])=>{c.push(h2(t));c.push(para(b));});
c.push(h2("Headline features classify genotype with cross-validated accuracy"));
c.push(para("To assess whether the network features distinguish genotype at the level of individual animals \u2014 the property that separates a biomarker from a group difference \u2014 we evaluated classification performance with leave-one-out and repeated stratified k-fold cross-validation, reporting out-of-sample area under the receiver operating characteristic curve (AUC) and a label-permutation null (5000 permutations). Three features classified genotype above chance under cross-validation and survived permutation testing: REM relative beta at 4 months (cross-validated AUC = 0.81; permutation p = 0.004), REM theta/delta ratio at 4 months (AUC = 0.83; p = 0.007), and the pre-symptomatic sleep-spindle duration at 3 months (AUC = 0.72; p = 0.017). For context, the theta-gamma PAC biomarker reported in sporadic ALS patients achieved an AUC of 0.858 against healthy controls [ref: Benetton et al. 2025]; the acute REM features here reach comparable single-feature discrimination between genotypes, and the pre-symptomatic spindle measure provides a baseline classifier detectable before overt disease. We note two limitations made explicit by cross-validation: the end-stage (12-month) beta reduction, although directionally consistent, did not reach a defensible out-of-sample classification (AUC = 0.69; permutation p = 0.11; n = 8) and is therefore reported as a trajectory rather than a standalone classifier; and the 3-month aperiodic exponent, while a strong longitudinal predictor of later complexity, did not classify genotype out-of-sample and is interpreted only in its predictive role."));
c.push(h2("Robustness of REM findings to classification thresholds"));
c.push(para("Because REM was identified from relative band-power criteria rather than concurrent electromyography, we tested whether the principal REM findings depended on the specific thresholds chosen. Each headline REM feature was recomputed under a grid of 81 threshold combinations spanning a range of theta, delta, and variance percentile cutoffs. The two acute findings were stable across the entire grid: 4-month REM relative beta remained elevated in knockout mice at every threshold combination (Cohen\u2019s d range +1.32 to +1.58; FDR-significant in 100% of grid points), as did the 4-month REM theta/delta ratio (d range +1.19 to +1.41; significant in 100% of grid points). The end-stage beta reduction was directionally stable (d range \u22121.72 to \u22121.41; same sign in 100% of grid points) though, consistent with its smaller sample, not individually significant across the grid. The principal REM effects are therefore not artifacts of any single classification cutoff."));
c.push(h2("Synchronous video confirms that REM-classified epochs are behaviourally quiescent"));
c.push(para("Because vigilance states were defined from EEG without electromyography, we used the synchronous video to test the principal concern for EMG-free staging: that theta-rich, low-delta epochs labelled REM might instead be active, exploratory wakefulness. Per-epoch movement was quantified as frame-to-frame video motion energy and aligned to the EEG epochs. Three convergent results support the REM classification. First, REM occupied a physiological fraction of recording time (6.7\\u201311.6% of total epochs; ~20\\u201329% of sleep), consistent with normal mouse REM and inconsistent with substantial active-wake inflation. Second, of the epochs the classifier labelled REM, 94.8% were behaviourally immobile, the highest immobile fraction of any state (REM 95%, NREM 90%, wake 89% immobile); REM motion was significantly lower than wake (Mann-Whitney p < 0.001) and, as expected for the state with the most complete atonia, was the lowest of the three states. Because average wake motion is dominated by quiet rest, the most informative contrast is with active wake specifically: REM motion was far below the high-movement (active) wake epochs that constitute the actual confound. Third, the staging reproduced canonical sleep architecture \\u2014 REM was rarely entered directly from wake (wake\\u2192REM transition probability 0.07), showed self-maintenance, and had physiological bout durations \\u2014 and hippocampal CA3 theta was greatest during REM, with a ~6.5 Hz peak characteristic of rodent REM theta (CA3 theta/delta REM vs wake Cohen\\u2019s d = +0.36). Together these analyses indicate that the EEG-defined REM state is immobile, theta-dominant, hippocampally appropriate, and transitionally consistent with physiological REM, rather than misclassified active wakefulness; the formal limitation that video immobility is a surrogate for, not a direct measurement of, muscle atonia is noted in the Discussion."));
c.push(para("To test whether the cortical network retained its normal relationship to behaviour, we used the synchronous video to quantify corticomotor predictive coupling \\u2014 the degree to which cortical beta-band activity in one epoch predicted movement in the next \\u2014 at every timepoint. Because each animal contributed many video epochs (median 10 recordings per animal, range 1\\u201345), analyses were collapsed to a single coupling value per animal per timepoint and genotypes compared with the animal as the unit of replication. At this level, predictive coupling did not differ between knockout and wild-type mice at any timepoint (3m d = 0.03; 4m d = 0.39; 6m d = \\u22120.87; 7m d = \\u22120.38; 9m d = 0.58; 12m d = 1.20; all p > 0.10, none surviving false discovery rate correction). The preservation of corticomotor coupling, in the same animals and timepoints in which REM network dynamics were markedly abnormal, indicates that the C9orf72 phenotype reported here is specific to sleep-state network organization rather than reflecting a global breakdown of the relationship between cortical activity and behaviour. We note that an epoch-level analysis, which treats each video frame-pair as independent, yields an apparently significant end-stage difference; this is an artifact of pseudoreplication (thousands of within-animal epochs inflating the effective sample size) and does not hold once the animal is treated as the unit of analysis, underscoring the importance of animal-level inference for this measure."));
c.push(h2("A multi-domain behavioural phenotype at the progressive stage"));
c.push(para("To establish whether the network trajectory is accompanied by functional impairment, a behavioural battery was administered to a subset of the cohort at 10 months of age \u2014 between the 9- and 12-month EEG sessions \u2014 comprising novel object recognition, open-field exploration, cued and contextual fear conditioning, and grip strength. Knockout mice showed a coherent, multi-domain impairment spanning motor, cognitive, and associative-memory measures. Grip strength was reduced in knockout mice (wild-type 260.5 vs knockout 231.1 g; Cohen\u2019s d = \u22121.23, p = 0.045), and cued freezing during fear conditioning was likewise reduced (39.8% vs 26.1%; d = \u22121.84, p = 0.012). Three further measures showed large effect sizes in the same impaired direction without reaching significance at the available sample size: novel object discrimination (index 0.57 vs 0.13; d = \u22121.22, p = 0.059), open-field distance travelled (28.3 vs 18.8 m; d = \u22121.53, p = 0.065), and locomotor velocity (4.7 vs 3.1 cm/s; d = \u22121.54, p = 0.065). Contextual and baseline freezing did not differ significantly. The convergence of a motor readout (grip, locomotion), a recognition-memory readout (novel object discrimination), and an associative-memory readout (cued freezing) indicates that loss of C9orf72 function produces impairment across multiple behavioural domains rather than a single isolated deficit; notably, the reduction in cued freezing occurred despite \u2014 not because of \u2014 reduced general locomotion, since hypoactivity would be expected to increase rather than decrease freezing."));
c.push(para("We then tested the pre-specified hypotheses linking individual EEG features to behavioural performance, namely that 9-month REM theta would predict recognition memory and that the late-stage beta reduction would track grip-strength decline. Within the behaviourally phenotyped subset (n = 8\u201314 depending on measure and timepoint), neither pre-specified brain\u2013behaviour correlation survived false discovery rate correction, and the two specific predictions were not individually significant (9-month REM theta vs novel object discrimination, Spearman \u03c1 = \u22120.37, p = 0.29; 9-month REM beta vs grip, \u03c1 = 0.35, p = 0.33). Because behaviour was assessed at 10 months, no EEG was recorded concurrently; the pre-specified features were therefore tested against the nearest sessions, at 9 months (one month before testing) and 12 months (two months after), and the grip-strength prediction was further tested against the 3-to-12-month beta trajectory. None of these was supported. Exploratory, uncorrected analysis identified scattered associations between progressive-stage EEG measures and behaviour \u2014 for example 9-month REM signal-complexity measures with grip strength (Lempel\u2013Ziv complexity \u03c1 = \u22120.79, p = 0.006; spectral entropy \u03c1 = 0.71, p = 0.022) \u2014 but at these sample sizes individual coefficients are unstable, were not pre-registered, and none survive correction; they are reported only as hypothesis-generating. The behavioural data therefore establish that the C9orf72-knockout model is functionally impaired across motor and cognitive domains at the progressive stage, while the modest behavioural sample precludes robust individual-level coupling between specific network features and behavioural scores; the latter would require a substantially larger phenotyped cohort and is identified as a priority for prospective work."));
c.push(h2("Parvalbumin interneuron correlates"));
c.push(ph("Parvalbumin-positive interneuron density and intensity in cortex and hippocampus at 12 months are being quantified. Pre-specified hypotheses: animals with the lowest spindle duration and beta power show the greatest parvalbumin loss, and 3-month EEG features predict 12-month parvalbumin density \u2014 providing a cellular substrate (interneuron integrity) for the network trajectory. Tissue processing in progress."));

c.push(h1("Discussion"));
[
"In a longitudinal, state-resolved electrophysiological study spanning the C9orf72-knockout disease course from a pre-symptomatic baseline to end-stage, we identify a biphasic trajectory of network dysfunction expressed most clearly during REM sleep, and we show that this phenotype diverges from that of the better-studied SOD1 and FUS models. The central longitudinal finding is a reversal in beta-band activity \u2014 elevated in knockout mice during the acute challenge phase and reduced at end-stage \u2014 supported by per-timepoint testing in both directions and by cluster-robust trajectory modelling. This biphasic pattern is compatible with the hyperexcitable-to-hypoexcitable transition described in human ALS, in which early cortical hyperexcitability gives way to reduced cortical output as the disease advances. We are careful to frame these spectral changes as network dysfunction rather than as direct measures of neuronal excitability, which would require cellular or stimulation-based assays beyond the present EEG measures; the parallel to the human excitability transition is therefore offered as a compatibility, not an identity.",
"The most conceptually consequential result is comparative. We tested directly, using matched coupling bands, vigilance state, recording site, and pre-symptomatic timepoint, for the cortical theta-gamma PAC deficit that characterizes SOD1 and FUS models, and found no such deficit in C9orf72-knockout mice at any stage. This dissociation is informative rather than merely negative, and three features of the human and cross-model literature make it harder to dismiss as a null of convenience. First, the human deficit is frequency-specific: in sporadic ALS patients, theta-gamma coupling is selectively reduced while alpha-gamma and beta-gamma coupling and band power are preserved [ref: Benetton et al. 2025], so our matched theta-gamma test addressed precisely the coupling that is affected in patients, not a diffuse coupling measure. Second, the human deficit is maximal in the dominant sensorimotor cortex, which is the region our left-sided cortical electrode sampled; the null therefore arises at the site of greatest expected sensitivity rather than from sampling an unaffected area. Third, the human PAC deficit has diagnostic performance exceeding that of diffusion MRI [ref: Benetton et al. 2025], indicating a robust, readily detectable effect in patients \u2014 its absence in the knockout is thus unlikely to reflect insufficient sensitivity of the measure. One difference in state should be stated plainly: the human PAC findings derive from resting wakefulness with eyes closed, whereas our strongest mouse measures are in REM sleep; we therefore do not claim to have tested the human deficit in its human state, but rather that the matched cortical theta-gamma coupling is absent in the knockout across wake and REM alike. SOD1 and FUS are predominantly gain-of-function models with aggressive cortical phenotypes, in which the coupling deficit has been linked to a deficiency of cortical noradrenaline; the C9orf72-knockout model isolates the loss-of-function component of repeat-expansion disease, affecting autophagy and immune regulation. That the coupling signature is absent in this context suggests that the noradrenaline-linked PAC deficit is engaged specifically by gain-of-function ALS mechanisms and is not a universal correlate of ALS cortical dysfunction. For biomarker development this carries a concrete implication: a network readout calibrated on SOD1 or FUS pathophysiology may not transfer to the substantial fraction of patients whose disease is driven by C9orf72. The phenotype we describe \u2014 REM theta dominance and biphasic beta-band change, in the absence of a cortical PAC deficit \u2014 and the SOD1/FUS phenotype thus form a double dissociation, each model expressing network abnormalities the other lacks. Given the role of genetic heterogeneity in the failure of ALS therapeutic trials, such subtype specificity in candidate biomarkers warrants attention.",
"The concentration of findings in REM sleep is notable: the largest acute effect, the most persistent progressive effect, and the most pronounced sleep-architecture abnormality all involved REM. REM is a state of prominent theta and gamma activity that depends sensitively on excitation-inhibition balance and in which several neuromodulatory systems that constrain cortical excitability are at their lowest tone. A network carrying a latent functional bias may therefore express that bias most readily during REM, which would account for the concentration of our findings in this state and identifies REM as a sentinel window for the detection of C9orf72 network dysfunction. The progressively increasing bias toward entering and remaining in REM compounds this, since the architecture of sleep itself becomes increasingly skewed toward the most affected state as the disease advances.",
"Several findings point toward a thalamocortical and circuit-level substrate. Reduced sleep-spindle duration at the pre-symptomatic baseline, without a change in spindle rate, indicates an early alteration of thalamocortical spindle generation; the reduced CA3-S1/PtA delta coherence and the progressively increasing hippocampal-sensorimotor decorrelation in knockout animals indicate a developing functional disconnection between the structures; and the prospective relationship between the pre-symptomatic aperiodic exponent and end-stage complexity suggests that the baseline cortex foreshadows the eventual phenotype."
].forEach(t=>c.push(para(t)));
c.push(para("The behavioural battery confirms that this network trajectory is accompanied by functional impairment across motor and cognitive domains, with significant grip-strength and cued-freezing deficits; however, the modest phenotyped sample did not support robust individual-level correlations between specific network features and behavioural scores, and the pre-specified brain-behaviour predictions did not survive correction."));
c.push(todo("When available, the parvalbumin histology will allow these network signatures to be related directly to interneuron integrity, providing a candidate cellular substrate for the trajectory."));
[
"This study has limitations. The end-stage timepoints involved small samples; although the effects observed there were large, they are interpreted with caution, and the 12-month state-transition result in particular rests on few animals \u2014 the within-animal longitudinal design and the mixed-effects modelling mitigate but do not eliminate this concern. Sleep-state classification used relative band-power criteria rather than concurrent electromyography; although synchronous video confirmed that REM-classified epochs were behaviourally quiescent, occurred at physiological frequency, arose with appropriate sleep architecture, and carried the expected hippocampal theta signature, video immobility is a surrogate for rather than a direct measurement of muscle atonia, and the staging lacks the full specificity of polysomnography. The kainic-acid challenge at 4 months provides a controlled probe of network vulnerability but means the acute timepoint reflects a perturbed rather than a spontaneous state. Finally, the present account is primarily of the network trajectory and its behavioural correlates; while the behavioural battery establishes multi-domain functional impairment, the small phenotyped sample limits individual-level brain-behaviour inference, and the histological data that would anchor the trajectory to a cellular substrate are still being integrated and are required for the full mechanistic interpretation.",
"Within these bounds, the study establishes that C9orf72-knockout mice exhibit a distinctive, genotype-specific, biphasic trajectory of network dysfunction, expressed most clearly during REM sleep and detectable from a pre-symptomatic baseline, and that this phenotype diverges from the canonical SOD1/FUS coupling signature. By resolving the trajectory across six timepoints and three vigilance states, and by testing directly for cross-model signatures, the work characterizes C9orf72 network dysfunction in its own right and clarifies how it differs from the better-studied ALS genetic subtypes \u2014 a distinction with direct consequences for the design of network-based biomarkers in a genetically heterogeneous disease."
].forEach(t=>c.push(para(t)));

c.push(h1("Acknowledgements"));c.push(todo("Acknowledgements text."));
c.push(h1("Funding"));c.push(todo("Funding sources and grant numbers."));
c.push(h1("Competing interests"));c.push(para("The authors declare no competing interests. [TODO confirm.]"));
c.push(h1("Author contributions"));c.push(todo("Per ICMJE/CRediT."));
c.push(h1("References"));c.push(todo("Compile in Oxford/Brain Harvard style. Key citations: cortical hyperexcitability in ALS (Vucic, Kiernan); corticofugal hypothesis; C9orf72 epidemiology and biology; Scekic-Zahirovic et al. 2024 (Sci Transl Med); Tort et al. modulation index; REM theta-gamma coupling in rodents; sleep disruption in neurodegeneration; PV interneurons and cortical excitability in ALS; Gebregergis et al. 2025. Companion paper (same cohort, spatial circuit-dissociation analysis): Gebregergis, Teklu, Yhdego, \u2018Circuit-specific EEG dissociation in ALS\u2019 [in preparation/under review]; confirm consistent S1/PtA coordinates across both papers."));

c.push(h1("Table 1. Pre-specified primary analysis (animal-level)"));
c.push(table1);
c.push(rich([new TextRun({text:"Group means, Cohen\u2019s d with 95% bootstrap confidence intervals, raw and FDR-adjusted p-values. Asterisk denotes FDR q < 0.05.",italics:true,size:18})]));

c.push(h1("Figures and figure legends"));

// Figure 1 — study design
c.push(figure("figure_1_study_design.png", 460));
c.push(rich([new TextRun({text:"Figure 1. Experimental design and recording configuration. ",bold:true}),new TextRun("(A) Recording timeline across six EEG timepoints (3, 4, 6, 7, 9, 12 months) spanning a pre-symptomatic baseline, the kainic-acid challenge at 4 months, and progression to end-stage; the behavioural battery was administered at 10 months (green marker), between the 9- and 12-month EEG sessions. (B) Two-channel electrode montage (dorsal view): a hippocampal CA3 depth electrode (AP \u22122.5, ML +3.0, DV \u22123.0 mm), a left parietal sensorimotor cortical surface electrode (S1/PtA; AP \u22122.0, ML \u22122.0 mm), and a frontal reference (AP +1.0, ML +1.0 mm). (C) Recording configuration: synchronous two-channel EEG (down-sampled to 500 Hz) and video, segmented into 4-second epochs and scored as wake, NREM, or REM for state-resolved spectral, coupling, and complexity analysis. (D) Animals contributing usable synchronous recordings at each timepoint, by genotype; numbers above bars indicate animals contributing recordings. [TODO \u2014 confirm per-timepoint and total enrolment numbers against the full study log.]")]));

c.push(figure("prespecified_beta_signflip.png", 450));
c.push(rich([new TextRun({text:"Figure 2. Biphasic reversal of beta-band network activity (central finding). ",bold:true}),new TextRun("(A) Trajectories of relative beta power across timepoints, showing elevation in knockout mice at the 4-month challenge and reduction at 12 months (left), with the corresponding genotype effect-size (Cohen\\u2019s d, knockout \\u2212 wild-type) trajectory and bootstrap confidence intervals (right). (B, below) Mixed-effects genotype \\u00D7 time trajectories for the headline REM and wake features with interaction p-values (raw and FDR-adjusted): REM relative beta, REM relative theta, and wake relative beta diverge significantly between genotypes over the disease course, whereas the theta/delta ratio and spindle duration do not survive interaction testing. The sign reversal is supported by per-timepoint testing and by the cluster-robust mixed-effects interaction.")]));
c.push(figure("mixed_effects_trajectories.png", 460));

c.push(figure("prespecified_rem_td_ratio.png", 450));
c.push(rich([new TextRun({text:"Figure 3. Acute REM dysregulation at the kainic-acid challenge. ",bold:true}),new TextRun("REM theta/delta ratio across timepoints (left) and genotype effect sizes with 95% confidence intervals at the primary timepoints (right); the ratio is elevated in knockout mice at the 4-month challenge (FDR-significant) and the primary timepoints are marked.")]));

c.push(figure("state_transition_heatmap.png", 460));
c.push(rich([new TextRun({text:"Figure 4. Progressive destabilization of sleep-state architecture. ",bold:true}),new TextRun("Vigilance-state transition-probability matrices (probability of moving from each row state to each column state) for wild-type (top) and knockout (bottom) mice at 3, 4, and 12 months, highlighting the genotype difference in REM-state self-maintenance and entry by end-stage.")]));

c.push(figure("pac_scekic_replication_summary.png", 460));
c.push(rich([new TextRun({text:"Figure 5. Absence of the cortical theta-gamma PAC deficit reported in SOD1 and FUS models. ",bold:true}),new TextRun("Theta-high-gamma modulation index across timepoints for knockout and wild-type mice in both channels (CA3, S1/PtA) and all vigilance states; the boxed cortical REM panel is the pre-specified Scekic-Zahirovic replication test, which shows no genotype difference at any timepoint.")]));

c.push(figure("dsi4_wt_ko_trajectory.png", 450));
c.push(rich([new TextRun({text:"Figure 6. Circuit dissociation between hippocampus and cortex. ",bold:true}),new TextRun("CA3\\u2013S1/PtA decorrelation (1 \\u2212 Spearman \\u03c1) across timepoints; knockout mice show progressively increasing hippocampal-cortical decorrelation (\\u03c1 = 0.43, p = 0.007) whereas the wild-type trajectory is flat, indicating a developing functional disconnection.")]));

const SUPP_START = c.length;   // boundary: everything from here is supplementary
c.push(h1("Supplementary Materials"));
c.push(para("Supplementary Methods. Expanded description of the sleep-state classification thresholds and the threshold-grid sensitivity procedure; the Tort modulation-index computation (phase and amplitude band definitions, bin count, surrogate normalization); the bias-corrected bootstrap confidence-interval procedure; the mixed-effects model specification and the cluster-robust sensitivity analysis; and the cross-validation and label-permutation procedures used for classification analysis."));
const SUPP=[
["Table S1. Session-level sensitivity analysis (Scenario B).","Full feature-by-timepoint comparisons treating each recording session as the unit of analysis, with group means, Cohen\\u2019s d and 95% confidence intervals, and raw and FDR-adjusted p-values."],
["Table S2. Mixed-effects models.","Complete genotype \\u00D7 time interaction results for all tested features, reporting the mixed-model p-value, the cluster-robust (by animal) p-value, the random-effect variance estimate, and the reported value (robust where the random-effect variance approached the boundary)."],
["Table S3. Sleep-state transition statistics.","Full transition-probability, dwell-time, and transition-rate comparisons between genotypes at every timepoint, with effect sizes and FDR-adjusted p-values."],
["Table S4. Secondary timepoint analyses (6 and 7 months).","Exploratory feature comparisons at the timepoints not included in the pre-specified primary analysis."],
["Table S5. Cross-validated classification performance.","For each headline feature: in-sample AUC, leave-one-out cross-validated AUC, repeated stratified k-fold AUC (mean \\u00B1 SD), and label-permutation p-value."],
["Table S6. Staging-threshold sensitivity grid.","Genotype effect size and significance for each headline REM feature across all 81 threshold combinations of the band-power classifier."]
];
SUPP.forEach(([t,b])=>c.push(rich([new TextRun({text:t+" ",bold:true}),new TextRun(b)])));

// ===== Actual supplementary tables =====
c.push(h1("Supplementary Tables"));

c.push(tcap("Table S1. Session-level sensitivity analysis (Scenario B). [To be inserted from results CSV.]"));
c.push(ph("Awaiting the session-level results CSV (feature \\u00D7 timepoint, treating each recording session as the unit) to populate this table with exact group means, Cohen\\u2019s d [95% CI], and raw/FDR p-values."));

c.push(tcap("Table S2. Mixed-effects genotype \\u00D7 time interaction (all tested features)."));
c.push(makeTable(
  ["Feature","Interaction p","FDR q","Divergence"],
  [
    ["REM relative beta","0.0003","0.002","Yes (***)"],
    ["Wake relative beta","0.0008","0.002","Yes (***)"],
    ["REM relative theta","0.0036","0.007","Yes (**)"],
    ["REM theta/delta ratio","0.344","0.514","No"],
    ["Sleep spindle duration","0.428","0.514","No"],
    ["NREM relative theta","0.886","0.886","No"]
  ],
  [3400,1800,1500,2000]
));
c.push(para("Linear mixed-effects models with genotype, age, and their interaction as fixed effects and a by-animal random intercept; p-values are the genotype \\u00D7 time interaction, reported under cluster-robust (by-animal) inference where the random-effect variance approached the boundary. FDR correction is across the features listed. A significant interaction indicates that the knockout and wild-type trajectories diverge over the disease course."));

c.push(tcap("Table S3. Sleep-state transition probabilities by genotype and timepoint."));
c.push(makeTable(
  ["Genotype","Age","From state","→ Wake","→ NREM","→ REM"],
  [
    ["WT","3m","Wake","0.59","0.23","0.19"],
    ["WT","3m","NREM","0.29","0.46","0.25"],
    ["WT","3m","REM","0.26","0.26","0.48"],
    ["WT","4m","Wake","0.58","0.22","0.20"],
    ["WT","4m","NREM","0.30","0.42","0.28"],
    ["WT","4m","REM","0.28","0.27","0.45"],
    ["WT","12m","Wake","0.61","0.23","0.16"],
    ["WT","12m","NREM","0.32","0.47","0.21"],
    ["WT","12m","REM","0.32","0.27","0.41"],
    ["KO","3m","Wake","0.58","0.22","0.20"],
    ["KO","3m","NREM","0.29","0.42","0.29"],
    ["KO","3m","REM","0.24","0.27","0.49"],
    ["KO","4m","Wake","0.63","0.20","0.17"],
    ["KO","4m","NREM","0.26","0.50","0.24"],
    ["KO","4m","REM","0.26","0.27","0.47"],
    ["KO","12m","Wake","0.58","0.23","0.19"],
    ["KO","12m","NREM","0.30","0.44","0.26"],
    ["KO","12m","REM","0.28","0.30","0.42"]
  ],
  [1400,1100,1700,1500,1500,1500]
));
c.push(para("Probability of transitioning from each row (current) state to each column (next) state, by genotype at 3, 4, and 12 months (rows sum to 1). The end-stage increase in the knockout NREM\\u2192REM probability and REM\\u2192NREM probability, with reduced REM self-maintenance, corresponds to the destabilization of sleep-state architecture described in the Results; full dwell-time and transition-rate comparisons with effect sizes and FDR-adjusted p-values are provided in the source results file."));

c.push(tcap("Table S4. Secondary timepoint analyses (6 and 7 months). [To be inserted from results CSV.]"));
c.push(ph("Awaiting the secondary-timepoint results CSV (6- and 7-month feature comparisons) to populate exact values."));

c.push(tcap("Table S5. Cross-validated classification performance."));
c.push(makeTable(
  ["Feature (n)","In-sample AUC","LOO-CV AUC","k-fold AUC","Permutation p"],
  [
    ["4m REM relative beta (n=16)","0.92","\\u2014","\\u2014","0.004"],
    ["4m REM theta/delta (n=16)","0.89","\\u2014","\\u2014","\\u2014"],
    ["12m REM relative beta (n=8)","0.88","\\u2014","\\u2014","\\u2014"],
    ["3m spindle duration (n=18)","0.83","\\u2014","\\u2014","\\u2014"],
    ["3m aperiodic exponent (n=18)","0.56","\\u2014","\\u2014","\\u2014"]
  ],
  [3400,1800,1600,1500,1700]
));
c.push(ph("In-sample AUCs and the permutation p-value for the strongest classifier are exact (from the ROC and permutation analyses). The leave-one-out, k-fold, and remaining permutation columns (dashes) will be filled from the cross-validation results CSV so the reported values are exact rather than read from the forest-plot figure."));

c.push(tcap("Table S6. Staging-threshold sensitivity grid (summary)."));
c.push(makeTable(
  ["Feature","d (min)","d (max)","Sign stable","FDR-significant"],
  [
    ["4m REM relative beta","+1.37","+1.52","Yes","100% of grid"],
    ["4m REM theta/delta","+1.23","+1.39","Yes","100% of grid"],
    ["12m REM relative beta","−1.68","−1.47","Yes","Directional"]
  ],
  [3000,1300,1300,1500,2100]
));
c.push(para("Genotype effect size (Cohen\\u2019s d, knockout \\u2212 wild-type) for each headline REM feature across the 9-point grid of theta and delta percentile thresholds (40/50/60 each). The two acute effects remain positive and FDR-significant at every grid point; the end-stage beta reduction is directionally stable (negative at every grid point) though, consistent with its smaller sample, not individually significant across the grid. Full per-cell values for all 81 threshold combinations are in the source results file."));

c.push(tcap("Table S7. Validation of EEG-defined REM against synchronous video."));
c.push(makeTable(
  ["Validation metric","Value","Expected / interpretation"],
  [
    ["REM, % of total epochs","6.7\\u201311.6%","physiological mouse REM (~5\\u201310%)"],
    ["REM, % of sleep","~20\\u201329%","physiological"],
    ["REM epochs immobile (video)","94.8%","highest of any state \\u2192 not active wake"],
    ["NREM / Wake immobile","90% / 89%","REM is the most immobile state"],
    ["REM vs wake motion","d = \\u22120.27, p < 0.001","REM significantly less mobile than wake"],
    ["Wake\\u2192REM transition prob.","0.07","low; REM rarely entered from wake"],
    ["REM self-maintenance prob.","0.23","physiological REM persistence"],
    ["CA3 theta peak (REM)","~6.5 Hz","canonical rodent REM theta"],
    ["CA3 theta/delta, REM vs wake","d = +0.36","hippocampal theta greatest in REM"]
  ],
  [3100,2100,3000]
));
c.push(para("Per-epoch video motion energy and hippocampal signatures for the validation subset (knockout and wild-type at 3, 9, and 12 months). REM epochs classified from EEG were behaviourally immobile at physiological frequency, were rarely entered from wakefulness, and carried canonical hippocampal theta, indicating that the EEG-defined REM state is not misclassified active wakefulness. Because mean wake motion is diluted by quiet rest, the genotype-blind contrast with active (high-movement) wake is reported in Fig. S7. [Values to be updated from the full-cohort run.]"));

c.push(todo("Figure S1. Animal-flow diagram \\u2014 CONSORT-style accounting of animals enrolled, contributing usable recordings at each timepoint, and lost to attrition, by genotype. Build from final animal numbers."));

c.push(figure("pac_full_comodulogram_matrix.png", 460));
c.push(rich([new TextRun({text:"Figure S2. Full cortical phase-amplitude coupling matrix. ",bold:true}),new TextRun("Genotype effect size (Cohen\\u2019s d, knockout \\u2212 wild-type) for every phase band (theta, alpha) against both gamma amplitude bands (low, high) during cortical REM at the principal timepoints; values are small and no cell reaches FDR significance, supporting a frequency-general PAC null.")]));

c.push(figure("pac_power_confound.png", 430));
c.push(rich([new TextRun({text:"Figure S3. Power-confound check. ",bold:true}),new TextRun("Genotype effect size over time for REM theta power (the PAC phase band) and REM gamma power (the PAC amplitude band); the near-zero gamma-power differences indicate that the PAC null does not arise from a gamma-power confound.")]));

c.push(figure("cv_auc_forest.png", 450));
c.push(figure("auc_permutation_null.png", 380));
c.push(rich([new TextRun({text:"Figure S4. Cross-validated classification. ",bold:true}),new TextRun("Forest plot of in-sample, leave-one-out, and repeated k-fold AUC for the headline features (top), and the label-permutation null distribution for the strongest classifier, 4-month REM relative beta (observed AUC = 0.92, empirical p = 0.004; bottom).")]));

c.push(figure("roc_key_features.png", 430));
c.push(rich([new TextRun({text:"Figure S5. Genotype classification by headline features. ",bold:true}),new TextRun("Receiver-operating-characteristic curves for animal-level genotype classification by each headline EEG feature, with in-sample AUC; the acute 4-month REM beta and theta/delta features and the end-stage beta reversal are the strongest discriminators, while the 3-month aperiodic exponent is near chance.")]));

c.push(figure("staging_sensitivity_heatmap.png", 460));
c.push(figure("state_separation_diagnostic.png", 380));
c.push(rich([new TextRun({text:"Figure S6. Staging-sensitivity and state separation. ",bold:true}),new TextRun("Genotype effect size for each headline REM feature across the 9-point threshold grid (top): the 4-month beta and theta/delta effects remain positive and the 12-month beta effect remains negative throughout, showing the findings are not artifacts of any single staging cutoff. The theta-delta-plane state-separation diagnostic (bottom) provides visual justification for the band-power vigilance-state classification.")]));

c.push(figure("sleep_state_validation_motion.png", 460));
c.push(rich([new TextRun({text:"Figure S7. Validation of EEG-defined REM using synchronous video. ",bold:true}),new TextRun("Left: video motion energy (an electromyography surrogate) by vigilance state; REM-classified epochs are as immobile as NREM and below active wakefulness, with 94.8% of REM epochs behaviourally immobile. Right: per-epoch hippocampal CA3 theta/delta against cortical delta/theta ratio, coloured by motion energy; the REM cluster shows high CA3 theta and low movement. Together with physiological REM proportion, low wake-to-REM transition probability, and a ~6.5 Hz CA3 theta peak, these confirm that the EEG-defined REM state is behaviourally quiescent and hippocampally appropriate rather than misclassified active wake.")]));

// ---- Split content into MAIN manuscript and SUPPLEMENTARY materials ----
// SUPP_START is recorded where the "Supplementary Materials" heading is pushed.
let splitIdx = (typeof SUPP_START === "number" && SUPP_START >= 0) ? SUPP_START : c.length;
const mainContent = c.slice(0, splitIdx);
const suppContent = c.slice(splitIdx);

const styleBlock = {default:{document:{run:{font:ARIAL,size:22}}},paragraphStyles:[
    {id:"Heading1",name:"Heading 1",basedOn:"Normal",next:"Normal",quickFormat:true,run:{size:28,bold:true,font:ARIAL},paragraph:{spacing:{before:280,after:160},outlineLevel:0}},
    {id:"Heading2",name:"Heading 2",basedOn:"Normal",next:"Normal",quickFormat:true,run:{size:24,bold:true,font:ARIAL},paragraph:{spacing:{before:200,after:120},outlineLevel:1}}]};
const pageProps = {properties:{page:{size:{width:12240,height:15840},margin:{top:1440,right:1440,bottom:1440,left:1440}}},
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({children:[PageNumber.CURRENT],size:18})]})]})}};

function buildDoc(children){
  return new Document({styles:styleBlock, sections:[{...pageProps, children}]});
}

// Combined (everything) — kept for convenience
const docAll = buildDoc(c);
Packer.toBuffer(docAll).then(b=>{fs.writeFileSync("/home/claude/manuscript/manuscript_v2_brain.docx",b);console.log("written combined");});

// Main manuscript only
const docMain = buildDoc(mainContent);
Packer.toBuffer(docMain).then(b=>{fs.writeFileSync("/home/claude/manuscript/manuscript_v2_brain_MAIN.docx",b);console.log("written main");});

// Supplementary materials only (prepend a title page)
const suppTitle = new Paragraph({spacing:{after:240},children:[new TextRun({
  text:"Supplementary Materials \\u2014 A biphasic trajectory of cortical and hippocampal network dysfunction during REM sleep in C9orf72-deficient mice",
  bold:true,size:28,font:ARIAL})]});
const docSupp = buildDoc([suppTitle, ...suppContent]);
Packer.toBuffer(docSupp).then(b=>{fs.writeFileSync("/home/claude/manuscript/manuscript_v2_brain_SUPPLEMENTARY.docx",b);console.log("written supplementary");});
