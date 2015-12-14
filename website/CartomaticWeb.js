$(document).ready(function(){
	$(".headerBtn").click(function(){
		var id = $(this).attr("href");
		$(".allDiv").not(id).hide();
		$(id).show();
	});
	
	window.setInterval(changeHomeImage, 4500);
	window.setInterval(changeAboutImage, 3500);
	
	//////////HOME
	
	var images = [], x = 0;
	images[0] = "image0.jpg";
	images[1] = "image1.jpg";
	images[2] = "image2.jpg";
	images[3] = "image3.jpg";
	
	function changeHomeImage(){
	$('#homeImage').hide();
	var currentImageString = "url("+images[x]+")";
	$('#homeImage').css("background-image", currentImageString);
	$('#homeImage').fadeIn(1500);
	x=x+1;
	if(x==4) x=0;
	}
	
	//////////ABOUT
	
	var Pimages = [], Px = 1;
	Pimages[0] = "p0.jpg";
	Pimages[1] = "p1.jpg";
	Pimages[2] = "p2.jpg";
	Pimages[3] = "p3.jpg";
	Pimages[4] = "p4.jpg";
	Pimages[5] = "p5.jpg";
	Pimages[6] = "p6.jpg";
	
	
	function changeAboutImage(){
	$('#aboutImage').hide();
	var currentImageString = "url("+Pimages[Px]+")";
	$('#aboutImage').css("background-image", currentImageString);
	$('#aboutImage').fadeIn(1500);
	
	var person = "#p"+Px;
	$(".members").not(person).css("color", "white")
	$(person).css("color", "rgb(236, 33, 231)");
	
	Px=Px+1;
	if(Px==7) Px=0;
	}
	
	
	
	changeHomeImage();
	changeAboutImage();
	
	
});

