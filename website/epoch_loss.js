// parte dello slider
var epoch_slider = document.getElementById("EpochLoss");
var displayEpoch = document.getElementById("displayEpochLoss");


var len=70;
var epoch=epoch_slider.value;
displayEpoch.innerHTML="epoch = "+epoch;

x_max=70;
y_max=1.5;
x_min=0;

var tau = (epoch)=>{
    return 1/60*Math.exp(-epoch/80)
}

const curve = (tau=1/20,x_0=1) =>{
    var out=[];
    var detail=300;
    for (let i=x_min/x_max*detail;i<=detail;i++){
        x=(i*x_max)/detail;
        out.push({x:x,y:x_0*Math.exp((x-60)*tau)});
    }
    out.push({x:x_max,y:0});
    return  out
}


var line=curve(epoch);

var document_width = parseInt(d3.select(".l-body").style("width"));

var height = 100;
var margin = {top: 20, right: 20, bottom: 30, left: 40};
var width = document_width - margin.left;

// append the svg object to the body of the page
var g0 = d3.select("#loss_graph")
    .append("g")
    .attr("width", width)
    .attr("height", height + margin.top + margin.bottom)
    .attr("transform","translate(" + margin.left + "," + margin.top + ")");


var x0 = d3.scaleLinear()
    .domain([0,1.1*x_max])
    .range([ 0, width ]);

    // Add Y axis
var y0 = d3.scaleLinear()
    .domain([0,1.5])
    .range([height, 0]);

const line0=d3.line()
    .x(function(d){return x0(d.x);})
    .y(function(d){return y0(d.y);})


// Add the line
g0.append("path")
    .data([line])
    .attr("d",line0)
    .attr("id","curve")
    .style("fill",'none')
    .style("stroke",'#298ec4')
    .style("stroke-width","3px");

g0.append("path")
    .data([curve(1/20,0.3)])
    .attr("d",line0)
    .attr("id","curve2")
    .style("fill",'none')
    .style("stroke",'#ff7300')
    .style("stroke-width","3px");
/*
vecchia versione del codice, funzionava solo con firefox.
Però era bella perchè aveva le cose formattate per benino

const syncR0 = function(){
    r = this.value;
    console.log(r);
    g0.select("#curve").data([curve(r)]).attr("d", line0);    
    displayR0.innerHTML="\\(r = "+r+"\\)";
    MathJax.typesetPromise([displayR0]);//slow
}

*/

g0.append("g")
    .attr("id","xaxis0")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x0));

g0.append("g")
    .attr("transform", "translate(0,0)")
    .call(d3.axisLeft(y0));


const syncEpoch = function(){
    epoch = epoch_slider.value;
    x0.range([0,width]);
    g0.select("#curve").data([curve(tau(epoch))]).attr("d", line0);
    g0.select("#curve2").data([curve(1/20,0.3)]).attr("d", line0);
    g0.select("#xaxis0")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x0));

    displayEpoch.innerHTML="epoch = "+epoch;  
}
syncEpoch();

const onresize = function(){
    document_width = parseInt(d3.select(".l-body").style("width"));
    width = document_width - margin.left;
    g0.attr("width", document_width - margin.left);
    syncEpoch();
    
}
onresize();

epoch_slider.addEventListener("input", syncEpoch);
window.addEventListener("resize", onresize);