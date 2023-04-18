//select styling
$('select').css('border', 'none');
//image styling
$('img').css('margin-left', 'auto');
$('img').css('margin-right', 'auto');
$('img').css('display', 'block');


/*
* Click detection
* */
$(document).ready(function(){
    $('#form').submit(function(event) {
        event.preventDefault();
        if($("#form").valid()){
            let fd = new FormData(event.currentTarget);
            $.ajax({
                url : '/calc/evaluate',
                type : 'POST',
                contentType: false,
                async: false,
                cache: false,
                processData: false,
                data : fd,
                success : function(data) {
                    let tag = $("#result");
                    let response = parseInt(data.code) == 1 ? "Good borrower" : "Bad borrower";
                    let response_color = parseInt(data.code) == 1 ? "bg-success" : "bg-danger";
                    tag.attr("class",response_color);
                    tag.html(response);
                },
                error : function(request,error) {
                    alert("Request: "+JSON.stringify(request));
                }
            });
        }
    });
});