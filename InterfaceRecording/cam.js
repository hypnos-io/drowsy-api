async function openCam() {
    let All_mediaDevices = navigator.mediaDevices
    if (!All_mediaDevices || !All_mediaDevices.getUserMedia) {
      console.log("getUserMedia() not supported.");
      return;
    }
    All_mediaDevices.getUserMedia({
      video: true,
    })
    .then(function(vidStream) {
      var video = document.getElementById('videoCam');
      if ("srcObject" in video) {
        video.srcObject = vidStream;
      } else {
        video.src = window.URL.createObjectURL(vidStream);
      }
      video.onloadedmetadata = function(e) {
        video.play();
        captureFrame(video);
      };
    })
    .catch(function(e) {
      console.log(e.name + ": " + e.message);
    });
  }

  var socket = io('http://localhost:3000');
  
  async function captureFrame(video) {
    var canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    var dataUrl = canvas.toDataURL();
    await socket.emit('process-image', { id: socket.id, images: [dataUrl] });
    setTimeout(function() {
      captureFrame(video);
    }, 1000); // capture a frame every 1000 milliseconds (1 second)
  }
  
