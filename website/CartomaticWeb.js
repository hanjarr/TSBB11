$(document).ready(function(){
	$(".headerBtn").click(function(){
		var id = $(this).attr("href");
		$(".allDiv").not(id).hide();
		$(id).show();
	});
	
	
	
	
	
});