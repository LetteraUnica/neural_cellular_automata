var colors=[["#fff","#fde411","#fd4949","#fef495","#fd9695","#fd8b1f"],   //base colors
            ["#fff","#fef0b0","#feb6b6","#fef9cc","#fecccc","#fec8b1"],   //dimemd colors colors
            ["#fff","#fde411","#fd4949","#fde411","#fd4949","#fd8b1f"]];  //bright colors
//var colors=[["#fff","#fde411","#fd4949","#fef495","#fd9695","#fd8b1f"],["#fff","#fdea7d","#fd8a8a","#fef6b1","#fdb2b1","#fdab7f"]];

// I like to log the data to the console for quick debugging

var image_width=parseInt(document.getElementById("grid_container").clientWidth)


console.log(image_width)
function gridData() {
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
				x: xpos,
				y: ypos,
				width: width,
				height: height,
				click: gridValue[row][column].click
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

var gridData = gridData();
console.log(gridData);

var article = document.getElementById("frame")
    .addEventListener('mouseover',function(){
        d3.selectAll(".square").style("fill",function (x){
            for (var i=0; i<6; i++){
                if (x.click==i) {return colors[0][i];}
            }
        })
        console.log('suca')
    })

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
    //.style("stroke", "#222")
	.style("fill", function(d) {
        for (var i=0; i<6;i++){
            if ((d.click)%6 == i ) { return colors[0][i]; }    //white
        }
    })
    .on('mouseover', function(d) { //d is the data of the square pointed with the mouse
        d3.selectAll(".square").style("fill",function (x){
            color_selected=d.click
            all_colors=x.click  //x is the data of all the squares
            if (d.click==0) {
                for (var i=0;i<6;i++){
                    if (x.click==i) {return colors[0][i];}
                }
            }    
            for (var i=0; i<6; i++){
                if (x.click==d.click) {return colors[2][color_selected];}
                if (x.click==i) {return colors[1][i];}
            }
        })
    });


