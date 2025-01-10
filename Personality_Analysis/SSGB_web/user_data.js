const chartData = {
    behavior: { labels: ['Night_owl', 'Early_bird', 'Decisive', 'Brand_loyalty'], data: [1640, 2408, 901, 5305] },
    favorite: { labels: ['Maker', 'Homebody', 'Culinarian', 'Geek', 'Photophile', 'Media_Aficionado', 'Audiophile', 'Fashionista', 'Lifestyle', 'Car_Enthusiast', 'Caregiver', 'Farm', 'Sport'], data: [2460, 5979, 11286, 7577, 363, 6383, 5884, 4855, 1677, 2949, 1476, 39, 700] },
    price: { labels: ['Common_Consumer', 'High_Consumer', 'Mid_Consumer'], data: [5918, 3707, 16432] },
};

const categoryDescriptions = {
    Night_owl: {
        description: 'Active at night, prefers evening activities.',
        traits: 'Creative, independent, often works or socializes late at night.',
        habits: 'Frequent use of social media or online platforms late at night.',
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
    Media_Aficionado: {
        description: 'Deeply engaged in various forms of media, including TV, movies, and books.',
        traits: 'Curious, informed, enjoys staying updated on trends.',
        habits: 'Spends on streaming services, movie tickets, and digital media subscriptions.',
        recommendations: 'Promote exclusive content, streaming bundles, or behind-the-scenes features.'
    },
    Audiophile: {
        description: 'Passionate about music and sound quality.',
        traits: 'Detail-oriented, tech-savvy, enjoys discovering new music.',
        habits: 'Purchases high-quality audio equipment, vinyl records, or music streaming subscriptions.',
        recommendations: 'Highlight premium headphones, lossless audio services, or curated playlists.'
    },
    Fashionista: {
        description: 'Keeps up with the latest trends in fashion and style.',
        traits: 'Trendy, expressive, enjoys experimenting with new looks.',
        habits: 'Frequent purchases of clothing, accessories, and beauty products.',
        recommendations: 'Promote seasonal collections, personalized styling services, or exclusive fashion events.'
    },
    Lifestyle: {
        description: 'Focuses on achieving a well-rounded and fulfilling life.',
        traits: 'Balanced, aspirational, values experiences and self-improvement.',
        habits: 'Spends on fitness, wellness products, and self-development courses.',
        recommendations: 'Highlight health subscriptions, mindfulness apps, or curated lifestyle boxes.'
    },
    Car_Enthusiast: {
        description: 'Loves cars and enjoys staying informed about the latest models and technologies.',
        traits: 'Mechanically inclined, adventurous, values performance and innovation.',
        habits: 'Buys car accessories, attends auto shows, and follows car-related media.',
        recommendations: 'Promote car care products, driving experiences, or automotive tech gadgets.'
    },
    Caregiver: {
        description: 'Takes care of family members or others in need.',
        traits: 'Empathetic, nurturing, values comfort and security for loved ones.',
        habits: 'Purchases health products, home care supplies, or educational materials.',
        recommendations: 'Highlight caregiving tools, family-focused services, or health support products.'
    },
    Farm: {
        description: 'Engaged in agriculture or values sustainable and organic living.',
        traits: 'Practical, environmentally conscious, enjoys nature.',
        habits: 'Buys farming tools, organic seeds, or sustainable products.',
        recommendations: 'Promote eco-friendly tools, gardening kits, or sustainable farming resources.'
    },
    Sport: {
        description: 'Passionate about sports, either as a participant or a fan.',
        traits: 'Active, competitive, enjoys teamwork and physical activity.',
        habits: 'Spends on sports equipment, team merchandise, or event tickets.',
        recommendations: 'Highlight performance gear, fitness trackers, or sports-related experiences.'
    },

    common_Consumer: {
        description: 'Represents the average consumer with diverse but general purchasing habits.',
        traits: 'Practical, value-conscious, seeks convenience and reliability.',
        habits: 'Spends on essential goods, general household items, and moderately priced services.',
        recommendations: 'Promote everyday products, value bundles, or widely appealing services.'
    },
    High_Consumer: {
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

const colors = [
    "#ff6384", // 粉红
    "#36a2eb", // 浅蓝
    "#cc65fe", // 紫色
    "#ffce56", // 柠檬黄
    "#4bc0c0", // 青绿
    "#ff9f40", // 橙色
    "#9966ff", // 浅紫
    "#c9cbff", // 淡蓝
    "#6bd4d6", // 浅青绿
    "#f7a8a8", // 淡粉
    "#ffcd94", // 浅橙黄
    "#d98cb3", // 浅玫红
    "#ffc658"  // 亮黄
];


let currentCategory = 'behavior';

// 函数：计算百分比
function calculatePercentage(data) {
    const total = data.reduce((sum, value) => sum + value, 0);
    return data.map(value => ((value / total) * 100).toFixed(2)); // 保留两位小数
}

const ctx = document.getElementById('chart').getContext('2d');
let myChart = new Chart(ctx, {
    type: 'pie',
    data: {
        labels: chartData[currentCategory].labels,
        datasets: [{
            label: 'Category Distribution',
            data: calculatePercentage(chartData[currentCategory].data),
            backgroundColor: colors,
        }],
    },
    options: {
        responsive: true,
        layout: {
            padding: {
                left: -30,
            },
        },
        plugins: {
            legend: {
                position: 'left',
            },
            tooltip: {
                callbacks: {
                    label: function (tooltipItem) {
                        const dataset = tooltipItem.dataset;
                        const dataIndex = tooltipItem.dataIndex;
                        const rawValue = dataset.data[dataIndex];

                        // 获取数据总和
                        const total = dataset.data.reduce((sum, value) => sum + parseFloat(value), 0);

                        // 计算百分比
                        const percentage = ((rawValue / total) * 100).toFixed(2);

                        // 获取标签
                        const label = dataset.label || '';
                        return `${tooltipItem.label}: ${percentage}%`;
                    },
                },
            },
        },
    }
});

function showLabels(category) {
    currentCategory = category;
    const data = chartData[category];
    myChart.data.labels = data.labels;
    myChart.data.datasets[0].data = data.data;
    myChart.update();
}

// 动态更新数据并重新渲染图表
function updateChartData(category, newData) {
    chartData[category].data = newData;
    const percentages = calculatePercentage(newData);
    myChart.data.labels = chartData[category].labels;
    myChart.data.datasets[0].data = percentages;
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
    const { jsPDF } = window.jspdf; // 从全局对象中获取 jsPDF
    const pdf = new jsPDF(); // 初始化 jsPDF

    try {
        const chartImage = myChart.toBase64Image(); // 获取图表为 Base64
        if (chartImage) {
            pdf.text('Clustering Results', 10, 10); // 添加标题
            pdf.addImage(chartImage, 'JPEG', 10, 20, 180, 120); // 添加图表
            pdf.text('Generated by Your App', 10, 150); // 添加注释
            pdf.save(`clustering_results_${new Date().toISOString().slice(0, 10)}.pdf`); // 保存 PDF
        } else {
            console.error('Chart image is not generated. Please check the chart rendering.');
            alert('Export fail. Please check the chart.');
        }
    } catch (error) {
        console.error('Export PDF failed:', error);
        alert('Export fail. Please check logbook for more info!');
    }
}




