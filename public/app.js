const docs=[
 {label:"support",text:"cannot sign in login password reset account locked"},
 {label:"support",text:"application crashes error loading page browser"},
 {label:"support",text:"help update profile change email address"},
 {label:"support",text:"website is slow and page will not open"},
 {label:"billing",text:"invoice payment card charged refund billing"},
 {label:"billing",text:"subscription renewal receipt payment failed"},
 {label:"billing",text:"update card details tax invoice charge"},
 {label:"billing",text:"refund request duplicate transaction payment"}
];
const tokenize=text=>(text.toLowerCase().match(/[a-z0-9]+/g)||[]);
function train(data){const labels=[...new Set(data.map(x=>x.label))],vocab=new Set(),stats={};for(const label of labels)stats[label]={docs:0,total:0,words:new Map()};for(const doc of data){const s=stats[doc.label];s.docs++;for(const word of tokenize(doc.text)){vocab.add(word);s.total++;s.words.set(word,(s.words.get(word)||0)+1);}}return {labels,vocab,stats,totalDocs:data.length};}
const model=train(docs);document.querySelector("#training").textContent=docs.map((d,i)=>(i+1)+". ["+d.label+"] "+d.text).join("\n");
function classify(text){const words=tokenize(text);const contributions={};for(const label of model.labels){const s=model.stats[label],prior=Math.log(s.docs/model.totalDocs),denom=s.total+model.vocab.size;const rows=words.map(word=>({word,value:Math.log(((s.words.get(word)||0)+1)/denom)}));contributions[label]=rows;contributions[label].score=prior+rows.reduce((sum,row)=>sum+row.value,0);contributions[label].prior=prior;}const max=Math.max(...model.labels.map(l=>contributions[l].score));const exps=model.labels.map(l=>Math.exp(contributions[l].score-max));const total=exps.reduce((a,b)=>a+b,0);const probabilities=Object.fromEntries(model.labels.map((l,i)=>[l,exps[i]/total]));const best=model.labels.reduce((a,b)=>probabilities[a]>=probabilities[b]?a:b);return {words,best,probabilities,contributions};}
document.querySelector("#classify").addEventListener("click",()=>{const text=document.querySelector("#text").value;const r=classify(text);document.querySelector("#answer").textContent=r.words.length?"Predicted category: "+r.best+" ("+(r.probabilities[r.best]*100).toFixed(1)+"% normalized score)":"Enter at least one word to classify.";document.querySelector("#scores").textContent=JSON.stringify(r.probabilities,null,2);document.querySelector("#tokens").textContent=JSON.stringify(r.contributions,null,2);});
document.querySelector("#classify").click();
