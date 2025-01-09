// 模拟的用户名和密码
const VALID_USERNAME = "admin";
const VALID_PASSWORD = "password";

// 登录功能
function login() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const errorMessage = document.getElementById('error-message');

    if (username === VALID_USERNAME && password === VALID_PASSWORD) {
        // 登录成功，跳转到主页面
        window.location.href = './index.html';
    } else {
        // 登录失败，显示错误消息
        errorMessage.textContent = "Invalid username or password!";
    }
}
