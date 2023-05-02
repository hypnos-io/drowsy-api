var http = require('http');
var fs = require('fs');
var index = fs.readFileSync('index.html');

var app = http.createServer(function (req, res) {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(index);
});

var io = require('socket.io').listen(app);

io.on('connection', function (socket) {
    console.log(`Socket conectado: ${socket.id}`)

    socket.on('sendFrame', function (data) {

        console.log(data);

    });

});

app.listen(3000);