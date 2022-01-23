var gridData = gridValue;	
var colors=[["#fff","#fde411","#fd4949","#fef495","#fd9695","#fd8b1f"],["#fff","#fff","#fff","#fff","#fff","#fff"]];

// I like to log the data to the console for quick debugging
console.log(gridData);

var grid = d3.select("#grid")
	.append("svg")
	.attr("width","100%")
	.attr("height","500px");
	
var row = grid.selectAll(".row")
	.data(gridData)
	.enter().append("g")
	.attr("class", "row");
	
var column = row.selectAll(".square")
	.data(function(d) { return d; })
	.enter().append("rect")
	.attr("class","square")
	.attr("x", function(d) { return d.x; })
	.attr("y", function(d) { return d.y; })
	.attr("width", function(d) { return d.width; })
	.attr("height", function(d) { return d.height; })
	.style("fill", function(d) {
        for (var i=0; i<6;i++){
            if ((d.click)%6 == i ) { return colors[0][i]; }    //white
        }
    })
    .on('mouseover', function(d) { //d is the data of the square pointed with the mouse
        d3.selectAll(".square").style("fill",function (x){
            color_selected=d.click%6
            all_colors=x.click%6
            if (color_selected==0) {
                for (var i=0;i<6;i++){
                    if (all_colors==i) {return colors[0][i];}
                }
            }    
            for (var i=0; i<6;i++){
                if (color_selected == i && color_selected!=all_colors) { return colors[1][i]; }
                if ((color_selected == i && color_selected==all_colors)) { return colors[0][i]; }
            }
        })
    });


	
function color_scheme(d){
    
}



/*
.on('mouseover', function(d) {
        d3.selectAll(".square").style("fill",function (x){
            for (var i=0; i<6;i++){
                if (d.click==0) { return colors[0][i]; }
                if ((d.click)%6 == i  && d.click!=x.click) { return colors[1][i]; } 
                if ((d.click)%6 == i  && d.click==x.click) {return colors[0][i];}   //white
            }
        })
    });
*/