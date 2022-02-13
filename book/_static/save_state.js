
/* This code is copied verbatim from this SO post by Rory McCrossan: https://stackoverflow.com/a/51543474/2217577.
The code was shared under the CC BY-SA 4.0 license: https://creativecommons.org/licenses/by-sa/4.0/ 
It's purpose is to simply store the state of checked boxes locally as a localStorage object.
To use it, simply add checkboxes as normal within your md files: 
<input type="checkbox" id="box-1" class="box"> Item 1 </input>
<input type="checkbox" id="box-2" class="box"> Item 2 </input>
<input type="checkbox" id="box-3" class="box"> Item 3 </input>
*/ 

function onClickBox() {
	var arr = $('.box').map(function() {
  	return this.checked;
  }).get();
  localStorage.setItem("checked", JSON.stringify(arr));
}

$(document).ready(function() {
	var arr = JSON.parse(localStorage.getItem('checked')) || [];
  arr.forEach(function(checked, i) {
  	$('.box').eq(i).prop('checked', checked);
  });

  $(".box").click(onClickBox);
});