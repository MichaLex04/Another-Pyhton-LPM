// Datos de las recetas
const recipes = {
    arroz: {
        name: "Arroz Blanco",
        ingredients: [
            { name: "Arroz", amount: 0.08, unit: "kg" },
            { name: "Agua", amount: 0.16, unit: "L" },
            { name: "Aceite", amount: 0.01, unit: "L" },
            { name: "Sal", amount: 0.002, unit: "kg" }
        ]
    },
    pollo: {
        name: "Pollo Guisado",
        ingredients: [
            { name: "Pollo", amount: 0.15, unit: "kg" },
            { name: "Cebolla", amount: 0.03, unit: "kg" },
            { name: "Tomate", amount: 0.04, unit: "kg" },
            { name: "Pimiento", amount: 0.02, unit: "kg" },
            { name: "Aceite", amount: 0.015, unit: "L" },
            { name: "Especias", amount: 0.005, unit: "kg" }
        ]
    },
    pasta: {
        name: "Pasta con Salsa",
        ingredients: [
            { name: "Pasta", amount: 0.1, unit: "kg" },
            { name: "Tomate", amount: 0.08, unit: "kg" },
            { name: "Cebolla", amount: 0.02, unit: "kg" },
            { name: "Ajo", amount: 0.005, unit: "kg" },
            { name: "Aceite de Oliva", amount: 0.015, unit: "L" },
            { name: "Albahaca", amount: 0.003, unit: "kg" }
        ]
    },
    ensalada: {
        name: "Ensalada Mixta",
        ingredients: [
            { name: "Lechuga", amount: 0.07, unit: "kg" },
            { name: "Tomate", amount: 0.05, unit: "kg" },
            { name: "Cebolla", amount: 0.02, unit: "kg" },
            { name: "Pepino", amount: 0.03, unit: "kg" },
            { name: "Aceite de Oliva", amount: 0.01, unit: "L" },
            { name: "Vinagre", amount: 0.005, unit: "L" }
        ]
    }
};

// Inicialización cuando el DOM esté cargado
document.addEventListener('DOMContentLoaded', function() {
    // Configurar eventos para las recetas
    const recipeItems = document.querySelectorAll('.recipe-item');
    recipeItems.forEach(item => {
        item.addEventListener('click', () => {
            recipeItems.forEach(i => i.classList.remove('selected'));
            item.classList.add('selected');
        });
    });

    // Configurar evento para el botón de cálculo
    document.getElementById('calculate-btn').addEventListener('click', calculateRations);

    // Calcular al cargar la página con valores por defecto
    calculateRations();
});

// Función principal de cálculo
function calculateRations() {
    // Mostrar loading
    document.getElementById('loading').style.display = 'block';
    
    // Simular procesamiento (en un caso real, aquí se conectaría con Python)
    setTimeout(() => {
        // Obtener valores de entrada
        const people = parseInt(document.getElementById('people').value);
        const lossPercentage = parseFloat(document.getElementById('lossPercentage').value);
        const fixedLoss = parseFloat(document.getElementById('fixedLoss').value);
        
        // Obtener receta seleccionada
        const selectedRecipe = document.querySelector('.recipe-item.selected').dataset.recipe;
        const recipe = recipes[selectedRecipe];
        
        // Calcular resultados
        let totalWeight = 0;
        let resultsHTML = `
            <h3>${recipe.name} para ${people} personas</h3>
            <p>Considerando ${lossPercentage}% de pérdidas y ${fixedLoss}kg de pérdida fija</p>
            <div class="ingredient-header">
                <strong>Ingrediente</strong>
                <strong>Cantidad Total</strong>
            </div>
        `;
        
        recipe.ingredients.forEach(ingredient => {
            // Calcular cantidad base
            let amount = ingredient.amount * people;
            
            // Aplicar porcentaje de pérdidas
            amount = amount * (1 + lossPercentage/100);
            
            // Aplicar pérdida fija para arroz (como en el ejemplo)
            if (selectedRecipe === 'arroz' && ingredient.name === 'Arroz') {
                amount += fixedLoss;
            }
            
            totalWeight += amount;
            
            // Formatear cantidad
            let formattedAmount;
            if (amount >= 1) {
                formattedAmount = amount.toFixed(2) + " " + ingredient.unit;
            } else {
                // Convertir a gramos o mililitros para cantidades pequeñas
                if (ingredient.unit === "kg") {
                    formattedAmount = (amount * 1000).toFixed(0) + " g";
                } else if (ingredient.unit === "L") {
                    formattedAmount = (amount * 1000).toFixed(0) + " ml";
                } else {
                    formattedAmount = amount.toFixed(2) + " " + ingredient.unit;
                }
            }
            
            resultsHTML += `
                <div class="ingredient-row">
                    <span class="ingredient-name">${ingredient.name}</span>
                    <span class="ingredient-amount">${formattedAmount}</span>
                </div>
            `;
        });
        
        // Actualizar la UI con los resultados
        document.getElementById('results').innerHTML = resultsHTML;
        document.getElementById('total-people').textContent = people;
        document.getElementById('total-loss').textContent = (fixedLoss + (totalWeight * lossPercentage/100)).toFixed(2) + " kg";
        document.getElementById('total-ingredients').textContent = totalWeight.toFixed(2) + " kg";
        
        // Ocultar loading
        document.getElementById('loading').style.display = 'none';
    }, 1000);
}

// Función para exportar resultados (extensión futura)
function exportResults(format) {
    alert(`Exportando resultados en formato ${format}. Esta función estará disponible pronto.`);
}

// Función para guardar receta (extensión futura)
function saveRecipe() {
    alert("Función de guardar receta estará disponible en la próxima versión.");
}