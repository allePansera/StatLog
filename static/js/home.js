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
        console.log($("#form").valid())
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
                    alert('Data: '+data.code);
                },
                error : function(request,error) {
                    alert("Request: "+JSON.stringify(request));
                }
            });
        }
    });
});