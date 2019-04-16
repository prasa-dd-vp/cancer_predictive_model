(function($) {
    "use strict"; // Start of use strict

    // jQuery for page scrolling feature - requires jQuery Easing plugin
    var params = {};
    $(document).on('click', 'a.page-scroll', function(event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: ($($anchor.attr('href')).offset().top - 50)
        }, 1250, 'easeInOutExpo');
        event.preventDefault();
    });

    $('#chat').click(function(){
        document.getElementById("chat").style.display = "none";  
        document.getElementById("myForm").style.display = "block";
    });

    $('#close').click(function(){
        document.getElementById("chat").style.display = "block";
        document.getElementById("myForm").style.display = "none";
    });

    $('#send').click(function(){
        var msg = document.getElementById('textarea').value;
        document.getElementById('textarea').value = null;
        sendtoDjango(msg);
    });

    $('.df').click(function(){
    var msg = this.innerHTML;
    sendtoDjango(msg);

});

    var sendtoDjango=function(data){
        $.ajax({
            type: "POST",// NO I18N
            url: "/home/request/", //NO I18N
            data: {'msg':data,
                    'intent':"nothing"},
            error: function() { 
                alert( 'some error occured' ); //NO I18N
            },
            success: function( resp ) {//alert(resp.Response);  
                $("#a").html(`<p>`+resp.Response+`</p>`); 
                button_generator (resp, data);     
            }
        })       
    } 

    $('#maritalStatus').hide();
    $('#race').hide();
    $('#primarySite').hide();
    $('#laterality').hide();
    $('#surgery').hide();

    var button_generator = function(resp, data){
            if (resp.intent === "Default Welcome Intent"){
                $('#maritalStatus').show();
                $('#race').hide();
                $('#primarySite').hide();
                $('#laterality').hide();
                $('#surgery').hide();

            }
            
            else if (resp.intent === "@details" && resp.Response === "Okay. May I know your age please"){
                params.maritalStatus = data;
                $('#maritalStatus').hide();
                $('#race').hide();
                $('#primarySite').hide();
                $('#laterality').hide();
                $('#surgery').hide();

            }
            else if (resp.intent === "@details" && resp.Response !== "Okay. May I know your age please"){
                params.age = data;
                $('#maritalStatus').hide();
                $('#race').show();
                $('#primarySite').hide();
                $('#laterality').hide();
                $('#surgery').hide();

            }
            else if(resp.intent==="@race"){
                params.race = data;
                $('#maritalStatus').hide();
                $('#race').hide();
                $('#primarySite').show();
                $('#laterality').hide();
                $('#surgery').hide();
            }
            else if(resp.intent==="@primary-site-number"){
                params.primarySite = data;
                $('#maritalStatus').hide();
                $('#race').hide();
                $('#primarySite').hide();
                $('#laterality').show();
                $('#surgery').hide();
            }
            else if(resp.intent==="@laterality"){
                params.laterality = data;
                $('#maritalStatus').hide();
                $('#race').hide();
                $('#primarySite').hide();
                $('#laterality').hide();
                $('#surgery').show();
            }
            else if(resp.intent==="@surgery"){
                params.surgery = data;
                $('#maritalStatus').hide();
                $('#race').hide();
                $('#primarySite').hide();
                $('#laterality').hide();
                $('#surgery').hide();
            }
            else if(resp.intent==="@tumor_marker_2"){
                $.ajax({
                type: "POST",// NO I18N
                url: "/home/predict/", //NO I18N
                data: params,
                error: function() { 
                    alert( 'hahaha' ); //NO I18N
                },
                success: function( resp ) {//alert(resp.Response);  
                    //alert(resp.asd);
                    $("#cs").html(resp.cs);
                    $("#sm").html(resp.sm);
                    
                }
            })    
            }
    }


    var add = (function () {
      var counter = 0;
      return function () {counter += 1; return counter;}
    })();

    


    // Highlight the top nav as scrolling occurs
    $('body').scrollspy({
        target: '.navbar-fixed-top',
        offset: 51
    });

    // Closes the Responsive Menu on Menu Item Click
    $('.navbar-collapse ul li a').click(function() {
        $('.navbar-toggle:visible').click();
    });

    // Offset for Main Navigation
    $('#mainNav').affix({
        offset: {
            top: 100
        }
    })

    // Initialize and Configure Scroll Reveal Animation
    window.sr = ScrollReveal();
    sr.reveal('.sr-icons', {
        duration: 600,
        scale: 0.3,
        distance: '0px'
    }, 200);
    sr.reveal('.sr-button', {
        duration: 1000,
        delay: 200
    });
    sr.reveal('.sr-contact', {
        duration: 600,
        scale: 0.3,
        distance: '0px'
    }, 300);

    // Initialize and Configure Magnific Popup Lightbox Plugin
    $('.popup-gallery').magnificPopup({
        delegate: 'a',
        type: 'image',
        tLoading: 'Loading image #%curr%...',
        mainClass: 'mfp-img-mobile',
        gallery: {
            enabled: true,
            navigateByImgClick: true,
            preload: [0, 1] // Will preload 0 - before current, and 1 after the current image
        },
        image: {
            tError: '<a href="%url%">The image #%curr%</a> could not be loaded.'
        }
    });

})(jQuery); // End of use strict
