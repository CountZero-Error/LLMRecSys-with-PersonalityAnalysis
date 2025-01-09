const chartData = {
    behavior: { labels: ['Night_owl', 'Early_bird', 'Decisive', 'Brand_loyalty'], data: [10, 20, 30, 40] },
    favorite: { labels: ['Maker', 'Homebody', 'Culinarian', 'Geek', 'Photophile'], data: [15, 25, 10, 5, 45] },
    price: { labels: ['high_consumer', 'Mid_Consumer'], data: [60, 40] },
};

const categoryDescriptions = {
    Night_owl: {
        description: 'Active at night, prefers evening activities.',
        traits: 'Creative, independent, often works or socializes late at night.',
        habits: 'Frequent use of social media or streaming platforms late at night.',
        recommendations: 'Promote late-night sales, offer nighttime delivery options, or highlight products like coffee and energy drinks.'
    },
    Early_bird: {
        description: 'Active in the morning, prefers early starts.',
        traits: 'Productive, health-conscious, values a structured routine.',
        habits: 'Often engages in morning workouts, prefers early shopping or dining.',
        recommendations: 'Promote early-morning discounts or breakfast items, and offer productivity tools.'
    },
    Decisive: {
        description: 'Makes decisions quickly.',
        traits: 'Goal-oriented, confident, less likely to overthink.',
        habits: 'Quickly selects products or services without much hesitation.',
        recommendations: 'Show clear product comparisons, highlight top-rated or best-seller items.'
    },
    Brand_loyalty: {
        description: 'Prefers sticking to trusted brands.',
        traits: 'Trustworthy, cautious, values consistency.',
        habits: 'Often repurchases from the same brand or product line.',
        recommendations: 'Offer loyalty rewards, exclusive brand-related promotions, or product upgrades.'
    },
    Maker: {
        description: 'Creative and enjoys building or crafting things.',
        traits: 'Innovative, hands-on, enjoys DIY projects.',
        habits: 'Frequent purchases of crafting supplies or tools.',
        recommendations: 'Promote DIY kits, online crafting tutorials, or customizable products.'
    },
    Homebody: {
        description: 'Prefers staying at home rather than going out.',
        traits: 'Comfort-seeking, introverted, enjoys cozy and familiar environments.',
        habits: 'Spends on home decor, loungewear, and streaming subscriptions.',
        recommendations: 'Promote home improvement products, comfortable clothing, or entertainment services.'
    },
    Culinarian: {
        description: 'Passionate about cooking and food.',
        traits: 'Creative, detail-oriented, enjoys experimenting with flavors.',
        habits: 'Purchases specialty ingredients, kitchen gadgets, or cookbooks.',
        recommendations: 'Highlight gourmet food products, cooking classes, or premium kitchen tools.'
    },
    Geek: {
        description: 'Enthusiastic about technology and sci-fi.',
        traits: 'Curious, analytical, enjoys learning about futuristic or technical topics.',
        habits: 'Buys tech gadgets, attends conventions, engages in online tech communities.',
        recommendations: 'Promote new gadgets, sci-fi collectibles, or tech-related workshops.'
    },
    Photophile: {
        description: 'Loves photography and capturing moments.',
        traits: 'Artistic, observant, enjoys traveling and documenting experiences.',
        habits: 'Invests in cameras, photo editing software, or travel accessories.',
        recommendations: 'Offer photography equipment, editing tools, or travel-related promotions.'
    },
    High_consumer: {
        description: 'Frequently purchases high-end products.',
        traits: 'Luxurious, status-conscious, appreciates premium quality.',
        habits: 'Shops for designer brands, high-tech gadgets, or exclusive services.',
        recommendations: 'Highlight luxury goods, limited-edition products, or personalized experiences.'
    },
    Mid_Consumer: {
        description: 'Prefers mid-range products.',
        traits: 'Practical, budget-conscious, values good quality at a reasonable price.',
        habits: 'Shops for reliable, well-reviewed items without overspending.',
        recommendations: 'Promote value-for-money products, bundles, or mid-tier brand collaborations.'
    }
};


let currentCategory = 'behavior';

const ctx = document.getElementById('chart').getContext('2d');
let myChart = new Chart(ctx, {
    type: 'pie',
    data: {
        labels: chartData[currentCategory].labels,
        datasets: [{
            label: 'Category Distribution',
            data: chartData[currentCategory].data,
            backgroundColor: ['#ff6384', '#36a2eb', '#cc65fe', '#ffce56', '#4bc0c0'],
        }],
    },
    options: {
        responsive: true,
        plugins: {
            tooltip: {
                callbacks: {
                    label: function (tooltipItem) {
                        const label = tooltipItem.label || '';
                        const value = tooltipItem.raw;
                        return `${label}: ${value}%`;
                    },
                },
            },
        },
    },
});

function showLabels(category) {
    currentCategory = category;
    const data = chartData[category];
    myChart.data.labels = data.labels;
    myChart.data.datasets[0].data = data.data;
    myChart.update();
}

// function searchUser() {
//     const userId = document.getElementById('user-id').value.trim();
//     if (userId) {
//         document.getElementById('user-details').innerText = `User ${userId}: Example tags - Night_owl, Brand_loyalty`;
//     } else {
//         document.getElementById('user-details').innerText = 'Please enter a valid User ID.';
//     }
// }

function closeDetails() {
    document.getElementById('details').style.display = 'none';
}

// 读取并解析 result.csv 文件，展示用户数据
function loadStaticData() {
    fetch('180_label.csv')
        .then(response => response.text())
        .then(data => {
            const rows = data.split('\n').slice(1); // 跳过CSV头
            const formattedData = rows.map(row => row.split(',')); // 按逗号分割每行
            renderUserData(formattedData); // 渲染到页面
        })
        .catch(error => {
            console.error('Error loading static file:', error);
            document.getElementById('static-data-display').innerText = 'Error loading data.';
        });
}

// 在页面上渲染用户数据
function renderUserData(userData) {
    const displayDiv = document.getElementById('static-data-display');
    displayDiv.innerHTML = ''; // 清空显示区域

    userData.forEach(([userId, labels]) => {
        if (userId && labels) { // 检查是否存在有效数据
            const userDiv = document.createElement('div');
            const tagList = labels.split('.').map(tag => `<span class="label">${tag}</span>`).join(', ');
            userDiv.innerHTML = `<strong>User ID:</strong> ${userId} <br> <strong>Labels:</strong> ${tagList}`;
            userDiv.className = 'user-entry';
            displayDiv.appendChild(userDiv);
        }
    });
}

// 用户搜索功能
function searchUser() {
    const userId = document.getElementById('user-id').value.trim();
    if (!userId) {
        document.getElementById('user-details').innerText = 'Please enter a valid User ID.';
        return;
    }

    fetch('180_label.csv')
        .then(response => response.text())
        .then(data => {
            const rows = data.split('\n').slice(1);
            const userData = rows.map(row => row.split(','));
            const user = userData.find(row => row[0] === userId);

            if (user) {
                const labels = user[1].split('.');
                const labelsHtml = labels.map(label => `<span class="label clickable-label" data-label="${label}">${label}</span>`).join(', ');
                document.getElementById('user-details').innerHTML = `
                    <strong>User ID:</strong> ${user[0]} <br>
                    <strong>Labels:</strong> ${labelsHtml}
                `;

                // 添加点击事件
                const labelElements = document.querySelectorAll('.clickable-label');
                labelElements.forEach(labelElement => {
                    labelElement.addEventListener('click', (event) => {
                        const label = event.target.getAttribute('data-label');
                        showCategoryDetails(label);
                    });
                });
            } else {
                document.getElementById('user-details').innerText = 'User not found.';
            }
        })
        .catch(error => {
            console.error('Error searching user:', error);
            document.getElementById('user-details').innerText = 'Error searching user data.';
        });
}

function showCategoryDetails(label) {
    label = label.trim();
    console.log('Label after cleaning:', label); // 调试输出清理后的label

    const details = categoryDescriptions[label]; // 匹配分类描述
    console.log('Category details:', details); // 调试是否找到匹配的描述

    if (details) {
        document.getElementById('details');
        const detailsElement = document.getElementById('details');
        console.log('Details visibility before:', detailsElement.style.display);
        detailsElement.style.display = 'block';
        detailsElement.offsetHeight;
        console.log('Details visibility after:', detailsElement.style.display);

        document.getElementById('category-description').innerHTML = `
            <strong>Description:</strong> ${details.description}<br>
            <strong>Traits:</strong> ${details.traits}<br>
            <strong>Habits:</strong> ${details.habits}<br>
            <strong>Recommendations:</strong> ${details.recommendations}
        `;
    } else {
        document.getElementById('category-description').innerText = 'No description available.';
    }
}




// 点击页面其他部分关闭详情框
document.addEventListener('click', function (event) {
    const detailsElement = document.getElementById('details');
    if (
        detailsElement.style.display === 'block' && // 如果详情框正在显示
        !detailsElement.contains(event.target) && // 并且点击的不是详情框内的内容
        !detailsElement.contains('clickable-label') &&
        event.target.id !== 'chart' // 并且点击的不是图
    ) {
        closeDetails();
    }
});


document.getElementById('chart').addEventListener('click', function (evt) {
    const points = myChart.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, true);
    if (points.length) {
        const index = points[0].index;
        const label = myChart.data.labels[index];
        const details = categoryDescriptions[label];

        if (details) {
            document.getElementById('details').style.display = 'block';
            document.getElementById('category-description').innerHTML = `
                <strong>Description:</strong> ${details.description}<br>
                <strong>Traits:</strong> ${details.traits}<br>
                <strong>Habits:</strong> ${details.habits}<br>
                <strong>Recommendations:</strong> ${details.recommendations}
            `;
        } else {
            document.getElementById('category-description').innerText = 'No description available.';
        }
    }
});


function exportPDF() {
    const pdf = new jsPDF();
    pdf.text('Clustering Results', 10, 10);
    pdf.addImage(myChart.toBase64Image(), 'JPEG', 10, 20, 180, 120); // 调整图表位置
    pdf.text('Generated by Your App', 10, 150); // 添加底部注释
    pdf.save(`clustering_results_${new Date().toISOString().slice(0, 10)}.pdf`); // 使用日期命名文件
}


