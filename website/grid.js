var colors=[["#fff","#fde411","#fd4949","#fef495","#fd9695","#fd8b1f"],   //base colors
            ["#fff","#fef0b0","#feb6b6","#fef9cc","#fecccc","#fec8b1"],   //dimmed colors colors
            ["#fff","#fde411","#fd4949","#fde411","#fd4949","#fd8b1f"]];  //bright colors

var explanation_colors=[["#fff","#fde411","#fd4949","#fff","#fff","#fff"],
                        ["#fff","#fde411","#fff","#fef495","#fff","#fff"],
                        ["#fff","#fde411","#fd4949","#fef495","#fff","#fef495"],
                        colors[0],
                        ["#fff","#fef0b0","#feb6b6","#fef9cc","#fecccc","#fd8b1f"]];


var text_ids=["white","yellow","red","light_yellow","light_red","orange"];

var image_width=parseInt(document.getElementById("grid_container").clientWidth)


console.log(image_width)
function gridData(gridFile) {
	var data = new Array();
	var xpos = 1; //starting xpos and ypos at 1 so the stroke will show when we make the grid below
	var ypos = 1;
	var width = Math.round(image_width/31);
	var height = width;
	var click = 0;
	
	// iterate for rows	
	for (var row = 0; row < 21; row++) {
		data.push( new Array() );
		
		// iterate for cells/columns inside rows
		for (var column = 0; column < 30; column++) {
			data[row].push({
                row: row,
                column: column,
				x: xpos,
				y: ypos,
				width: width,
				height: height,
				click: gridFile[row][column].click
			})
			// increment the x position. I.e. move it over by 50 (width variable)
			xpos += width;
		}
		// reset the x position after a row is complete
		xpos = 1;
		// increment the y position for the next row. Move it down 50 (height variable)
		ypos += height;	
	}
	return data;
}


//this hightlights the everything associated to a certain id
hightlight = function(id){
    for (var i = 0; i < 6; i++) { //this for is for the caption
        var display="none";
        if (i==id){ display="block";}
        document.getElementById(text_ids[i]).style.display=display;
    }
    d3.selectAll(".square").style("fill",function (x){//x is the data of all the squares
        if (id==0) { //this is when you highlight all the squares
            return colors[0][x.click];
        }
        if (x.click==id) {return colors[2][id];} //this is when you highlight a specific square
        return colors[1][x.click];              //this to dimm all the other squares
        
    })
}





var gridData = gridData(gridValue);



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
    .attr("row",function(d){return d.row;})
    .attr("column",function(d){return d.column;})
	.attr("x", function(d) { return d.x; })
	.attr("y", function(d) { return d.y; })
	.attr("width", function(d) { return d.width; })
	.attr("height", function(d) { return d.height; })
    //.style("stroke", "#222")
	.style("fill", function(d) {
        for (var i=0; i<6;i++){
            if ((d.click)%6 == i ) { return colors[0][i]; }    //white
        }
    })
    .on('mouseover', function(d) { //d is the data of the square pointed with the mouse
        hightlight(d.click);
    });


console.log(gridYellowValue);

var explanation_ids=["explanation 1","explanation 2","explanation 3","explanation 4","explanation 5"];


explanation_hightlight = function(id){
    d3.selectAll(".square").style("fill",function (x){//x is the data of all the squares
        for(var i=0;i<5;i++){
            var color="grey";
            if (id==i) {color="black";}
            document.getElementById(explanation_ids[i]).style.color=color;
        }
        var grid= gridValue;
        if (id==1){grid=gridYellowValue;}
        var click = grid[x.row][x.column].click%6;
        if (click==-1){return colors[0][click];}
        return explanation_colors[id][click];             
        
    })
}


let explanation = new Array(6);
for (var i=0;i<5;i++){ //this is for the explanation on the right
    explanation[i]=document.getElementById(explanation_ids[i])
    explanation[i].onmouseover=function(){
        explanation_hightlight(explanation_ids.indexOf(this.id));
    };
}



//this is for resetting to default when the mouse is far from the image
var article = document.getElementById("frame")
    .addEventListener('mouseover',function(){
        hightlight(0);
        explanation_hightlight(-1);
    })