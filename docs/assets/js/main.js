$(document).ready(function() {
	
    var elements = $('.sticky');
    Stickyfill.add(elements);



    $('body').scrollspy({target: '#doc-menu', offset: 100});
    
   
	$('a.scrollto').on('click', function(e){
       
        var target = this.hash;    
        e.preventDefault();
		$('body').scrollTo(target, 800, {offset: 0, 'axis':'y'});
		
	});
     
 

      


});