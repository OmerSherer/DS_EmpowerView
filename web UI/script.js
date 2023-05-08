const cameraButton = document.getElementById('camera');
const closeCameraButton = document.getElementById('close-camera');
const videoStream = document.getElementById('video-stream');
const fileInput = document.getElementById('file-upload');
const uploadButton = document.getElementById('upload-button');

cameraButton.onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
    videoStream.srcObject = stream;
    videoStream.play();
    cameraButton.style.display = 'none';
    closeCameraButton.style.display = 'block';
    uploadButton.style.display = 'none';
}

closeCameraButton.onclick = () => {
    videoStream.srcObject.getTracks()[0].stop();
    videoStream.srcObject = null;
    closeCameraButton.style.display = 'none';
    cameraButton.style.display = 'block';
    if (fileInput.value) {
        uploadButton.style.display = 'block';
    }
}

fileInput.onchange = () => {
    if (fileInput.value) {
        uploadButton.style.display = 'block';
    } else {
        uploadButton.style.display = 'none';
    }
}

uploadButton.onclick = () => {
    alert('Upload video successfully');
    fileInput.value = '';
    uploadButton.style.display = 'none';
}
