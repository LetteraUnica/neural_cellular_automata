//source

function gridData() {
	var data = new Array();
	var xpos = 1; //starting xpos and ypos at 1 so the stroke will show when we make the grid below
	var ypos = 1;
	var width = 15;
	var height = 15;
	var click = 0;
	
	// iterate for rows	
	for (var row = 0; row < 22; row++) {
		data.push( new Array() );
		
		// iterate for cells/columns inside rows
		for (var column = 0; column < 31; column++) {
			data[row].push({
				x: xpos,
				y: ypos,
				width: width,
				height: height,
				click: click
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
// I like to log the data to the console for quick debugging
console.log(gridData);

var grid = d3.select("#grid")
	.append("svg")
	.attr("width","510px")
	.attr("height","510px");
	
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
	.style("fill", "#fff")
	.style("stroke", "#222")
	.on('click', function(d) {
       d.click ++;
       if ((d.click)%6 == 0 ) { d3.select(this).style("fill","#fff"); } 
	   if ((d.click)%6 == 1 ) { d3.select(this).style("fill","#fde411"); } //yellow
	   if ((d.click)%6 == 2 ) { d3.select(this).style("fill","#fd4949"); } //red
	   if ((d.click)%6 == 3 ) { d3.select(this).style("fill","#fef495"); } //light-yellow
	   if ((d.click)%6 == 4 ) { d3.select(this).style("fill","#fd9695"); } //light-red
	   if ((d.click)%6 == 5 ) { d3.select(this).style("fill","#fd8b1f"); } //orange

    });

	
