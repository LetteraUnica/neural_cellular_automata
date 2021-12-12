



const square = function(x_min,x_max,y_max){
   return [{x:x_min,y:0},{x:x_min,y:y_max},{x:x_max,y:y_max},{x:x_max,y:0}];
}



// append the svg object to the body of the page
var g1 = d3.select("#loss_graph_growing")
    .append("svg")
    .attr("width", width)
    .append("g")
    .attr("height", height + margin.top + margin.bottom)
    .attr("transform","translate(" + margin.left + "," + margin.top + ")");

var x1 = d3.scaleLinear()
    .domain([0,200])
    .range([ 0, width ]);


const line1=d3.line()
    .x(function(d){return x1(d.x);})
    .y(function(d){return y0(d.y);})



// Add the line
g1.append("path")
    .data([curve(-1/60,x_0=1,x_max=96,x_min=64,out=[{x:64,y:0}])])
    .attr("d",line1)
    .attr("id","curve_growing")
    .style("fill",'none')
    .style("stroke",'#298ec4')
    .style("stroke-width","3px");

g1.append("path")
    .data([square(160,280,0.5)])
    .attr("d",line1)
    .attr("id","curve_square")
    .style("fill",'none')
    .style("stroke",'#ff7300')
    .style("stroke-width","3px");


g1.append("g")
    .attr("id","xaxis1")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x1));

g1.append("g")
    .attr("transform", "translate(0,0)")
    .call(d3.axisLeft(y0));



const onresize2 = function(){
    document_width = parseInt(d3.select(".l-body").style("width"));
    width = document_width - margin.left;
    g1.attr("width", document_width - margin.left);
    x0.range([0,width]);
    g1.select("#curve_growing").attr("d", line1);
    g1.select("#curve_square").attr("d", line1);
    g1.select("#xaxis1")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x1));
}
onresize2();

