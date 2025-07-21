document.addEventListener("DOMContentLoaded", () => {
	const outputBox = document.getElementById("output-box");
	const inputBox = document.getElementById("input-box");
	const phraseTable = document.getElementById("phrase-table");
	const phraseButton = document.getElementById("phrase-button");
	const tokenButton = document.getElementById("token-button");
	const allButton = document.getElementById("all-button");
	const generationBox = document.getElementById("generation-box");
	// Panels
	const outlinePanel = document.getElementById("outline-panel");
	const alterPanel = document.getElementById("alternatives-panel");
	const alterTokenPanel = document.getElementById("alternatives-token-panel");
	const alterPhrasePanel = document.getElementById("alternatives-phrase-panel");
	const alterTitle = document.getElementById("alternatives-title");
	const generationButton = document.getElementById("generate-button");
	const exampleButton = document.getElementById("example-button");

	// --- STATE ---
	let currentTokenSequence = [];
	let currentlySelectedTokenSpan = null;

	/**
	 * 将一个包含多种分隔符的字符串转换为一个干净的字符串数组。
	 * 分隔符可以是：逗号(,), 制表符(\t), 换行符(\n)。
	 *
	 * @param {string} phrases - 包含短语的原始字符串。
	 * @returns {string[]} - 清理和分割后的短语数组。
	 */
	function splitPhrases(phrases) {
		if (typeof phrases !== "string" || phrases.length === 0) {
			return [];
		}
		const phrasesArray = phrases
			.split(/[,\t\n]+/)
			.map((phrase) => phrase.trim())
			.filter((phrase) => phrase.length > 0);

		return phrasesArray;
	}

	async function FetchTokensFromServer(prefix) {
		// Clear the alternativesPanel
		alterTitle.innerHTML = "DVA is generating response...";
		alterTitle.style.color = "#278fc7";
		while (alterPanel.childElementCount > 1) {
			alterPanel.removeChild();
		}
		// Phrases Table
		let phrases = splitPhrases(phraseTable.value);
		console.log("Phrases:", phrases);

		const CSRFToken = document.querySelector(
			"[name=csrfmiddlewaretoken]"
		).value;

		// Fetch suffix from backend
		try {
			let response = await fetch("/generation/generate/", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					"X-CSRFToken": CSRFToken,
				},
				body: JSON.stringify({ prefix: prefix, phrases: phrases }),
			});

			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			response = await response.json();
			results = response.results;
			console.log("Suffix:", results.map((t) => t.chosenToken).join(""));
			return results;
		} catch (error) {
			console.error("Error:", error);
		}
		return null;
	}

	function AlternativeBox(token, percentage, type, selectedTokenIndex) {
		// <div class="box option-box">
		const optionBox = document.createElement("div");
		optionBox.classList.add("box", "option-box");

		// highlight the selected token
		if (token == currentTokenSequence[selectedTokenIndex].chosenToken) {
			optionBox.classList.add("option-selected");
		}

		// <div class="option-textarea">
		const optionTextarea = document.createElement("div");
		optionTextarea.classList.add("option-textarea");

		// <p class="option-token">
		const optionToken = document.createElement("p");
		optionToken.classList.add("option-token");
		optionToken.textContent = token;
		if (type == "token") {
			optionToken.classList.add("token");
		} else {
			optionToken.classList.add("phrase");
		}

		// <p class="option-percent">
		const optionPercent = document.createElement("p");
		optionPercent.classList.add("option-percent");
		if (percentage > 0.0001) {
			optionPercent.textContent = `${percentage.toFixed(3)}%`;
		} else {
			optionPercent.textContent = `${percentage.toExponential(3)}%`;
		}

		// <progress></progress>
		const progressBar = document.createElement("progress");
		progressBar.classList.add("progress", "is-success", "option-progress");
		progressBar.setAttribute("value", percentage);
		progressBar.setAttribute("max", "100");

		// Assembly Elements
		optionTextarea.appendChild(optionToken);
		optionTextarea.appendChild(optionPercent);
		optionBox.appendChild(optionTextarea);
		optionBox.appendChild(progressBar);

		// Add listener
		optionBox.addEventListener("click", async () => {
			await handleAlternativeSelection(selectedTokenIndex, token);
		});

		return optionBox;
	}

	function renderCurrentTokens(selectedTokenIndex) {
		// Change the outputbox and alternativebox style
		alterTitle.innerHTML = "Click token above to see alternatives.";
		alterTitle.style.fontWeight = "bold";
		alterTitle.style.color = "#32d6e2";

		if (currentlySelectedTokenSpan) {
			currentlySelectedTokenSpan.classList.remove("selected-for-alternatives");
			currentlySelectedTokenSpan = null;
		}

		const typingSpeed = 150; // 每个词元出现的间隔时间（毫秒）
		let tokenAddIndex = 0; // 用于计算递增延迟的计数器

		// Add the generated tokens to output box
		currentTokenSequence.forEach((tokenData, index) => {
			if (index > selectedTokenIndex) {
				const tokenSpan = document.createElement("span");
				tokenSpan.classList.add("token-span");
				if (tokenData.type == "token") {
					tokenSpan.classList.add("token");
				} else {
					tokenSpan.classList.add("phrase");
				}
				tokenSpan.textContent = tokenData.chosenToken;
				tokenSpan.dataset.index = index;

				tokenSpan.addEventListener("click", (event) => {
					event.stopPropagation();
					const clickedIndex = parseInt(tokenSpan.dataset.index);

					// refresh the selected token
					if (currentlySelectedTokenSpan) {
						currentlySelectedTokenSpan.classList.remove(
							"selected-for-alternatives"
						);
					}

					tokenSpan.classList.add("selected-for-alternatives");
					currentlySelectedTokenSpan = tokenSpan;

					// display the alternative tokens
					displayAlternatives(
						currentTokenSequence[clickedIndex].alternatives,
						clickedIndex
					);
				});

				// outputBox.appendChild(tokenSpan);

				// 安排此词元在延迟后出现
				setTimeout(() => {
					outputBox.appendChild(tokenSpan);
				}, tokenAddIndex * typingSpeed);

				tokenAddIndex++; // 增加计数器，以便下一个词元有更长的延迟
			}
		});
	}

	// [Alternative Display Event] Added to TokenSpan
	function displayAlternatives(alternatives, selectedTokenIndex) {
		console.log(currentTokenSequence[selectedTokenIndex].chosenToken);
		console.log(alternatives);

		// Clear and display alternativePanel
		ClearAlternatives();
		phraseButton.click();

		// Display the outline
		outlinePanel.style.display = "block";
		// Display the three buttons
		tokenButton.style.display = "inline-flex";
		phraseButton.style.display = "inline-flex";
		allButton.style.display = "inline-flex";

		// Modify title for alternative box
		let currentChosenTokenText =
			currentTokenSequence[selectedTokenIndex].chosenToken;
		alterTitle.textContent = `Current selected token: "${currentChosenTokenText}"`;

		// Add optionBox to alternativesPanel
		alternatives.forEach((alt) => {
			alterPanel.appendChild(
				AlternativeBox(alt.token, alt.prob * 100, alt.type, selectedTokenIndex)
			);
			if (alt.type == "token") {
				alterTokenPanel.appendChild(
					AlternativeBox(
						alt.token,
						alt.prob * 100,
						alt.type,
						selectedTokenIndex
					)
				);
			}
			if (alt.type == "phrase") {
				alterPhrasePanel.appendChild(
					AlternativeBox(
						alt.token,
						alt.prob * 100,
						alt.type,
						selectedTokenIndex
					)
				);
			}
		});
	}

	async function handleAlternativeSelection(selectedTokenIndex, selectedToken) {
		// Hide the alternatives panel
		generationButton.disabled = true;
		ClearAlternatives();

		// 1. Construct the prefix, containing the selectedToken
		let prefix = currentTokenSequence
			.slice(0, selectedTokenIndex + 1)
			.map((td) => td.chosenToken)
			.join("");
		prefix = prefix.concat(selectedToken);
		console.log(`Received Prefix: ${prefix}`);

		// 2. Change the outputBox and alternativePenal style, Remove the suffix tokens from output box
		outputBox.style.opacity = "0.6";
		alterTitle.innerHTML = "Generate new sequence...";
		currentTokenSequence[selectedTokenIndex].chosenToken = selectedToken;
		let selectedTokenItem = outputBox.querySelector(
			`:nth-child(${selectedTokenIndex + 1})`
		);
		selectedTokenItem.innerHTML = selectedToken;

		console.log(outputBox.childElementCount);
		console.log(selectedTokenIndex);
		for (
			let idx = outputBox.childElementCount;
			idx > selectedTokenIndex + 1;
			idx--
		) {
			outputBox.removeChild(outputBox.lastChild);
		}

		// 2. Get the new suffix
		const newSuffixTokensData = await FetchTokensFromServer(prefix); // generation
		let newSequence = currentTokenSequence.slice(0, selectedTokenIndex + 1);
		console.log("Prefix Array");
		console.log(newSequence);
		currentTokenSequence = newSequence.concat(
			newSuffixTokensData.map((item, idx) => ({
				...item,
				originalIndex: selectedTokenIndex + idx,
			}))
		);
		console.log("New Array");
		console.log(currentTokenSequence);

		// 3. Render the new Suffix
		renderCurrentTokens(selectedTokenIndex);

		// 4. Reset the outputBox and alternativePenal style
		generationButton.disabled = false;
		outputBox.style.opacity = "1";
	}

	// Hide the buttons and panels
	function ClearAlternatives() {
		outlinePanel.style.display = "none";
		alterPanel.innerHTML = "";
		alterPhrasePanel.innerHTML = "";
		alterTokenPanel.innerHTML = "";
		alterPanel.style.display = "none";
		alterPhrasePanel.style.display = "none";
		alterTokenPanel.style.display = "none";
		tokenButton.style.display = "none";
		phraseButton.style.display = "none";
		allButton.style.display = "none";
	}

	// Generate button click event
	generationButton.addEventListener("click", async () => {
		// Clean
		currentlySelectedTokenSpan = null;
		currentTokenSequence = [];
		outputBox.innerHTML = "";
		// Generation
		generationButton.disabled = true;
		exampleButton.disabled = true;

		let prefix = inputBox.value;
		console.log(`Init Prefix: ${prefix}`);
		let TokensData = await FetchTokensFromServer(prefix);
		currentTokenSequence = TokensData.map((item, index) => ({
			...item,
			originalIndex: index,
		}));
		renderCurrentTokens(-1);
		generationButton.disabled = false;
		exampleButton.disabled = false;
	});

	// Example button click event
	exampleButton.addEventListener("click", async () => {
		inputBox.value = "Introduce China to me:";
		phraseTable.value = "China, Chinese, Beijing, Shanghai, Guangzhou";
		generationButton.click();
	});

	// Phrase button click event
	inputBox.addEventListener("keydown", function (event) {
		if (event.isComposing) {
			return;
		}
		if (event.key === "Enter" && !event.shiftKey) {
			event.preventDefault();
			generationButton.click();
		}
	});

	// region Three Buttons Click Events
	phraseButton.addEventListener("click", () => {
		alterPanel.style.display = "none";
		allButton.classList.add("is-link");
		allButton.classList.add("is-light");
		allButton.classList.remove("is-info");
		alterPhrasePanel.style.display = "block";
		phraseButton.classList.remove("is-link");
		phraseButton.classList.remove("is-light");
		phraseButton.classList.add("is-info");
		alterTokenPanel.style.display = "none";
		tokenButton.classList.add("is-link");
		tokenButton.classList.add("is-light");
		tokenButton.classList.remove("is-info");
	});

	allButton.addEventListener("click", () => {
		alterPanel.style.display = "block";
		allButton.classList.remove("is-link");
		allButton.classList.remove("is-light");
		allButton.classList.add("is-info");
		alterPhrasePanel.style.display = "none";
		phraseButton.classList.add("is-link");
		phraseButton.classList.add("is-light");
		phraseButton.classList.remove("is-info");
		alterTokenPanel.style.display = "none";
		tokenButton.classList.add("is-link");
		tokenButton.classList.add("is-light");
		tokenButton.classList.remove("is-info");
	});

	tokenButton.addEventListener("click", () => {
		alterPanel.style.display = "none";
		allButton.classList.add("is-link");
		allButton.classList.add("is-light");
		allButton.classList.remove("is-info");
		alterPhrasePanel.style.display = "none";
		phraseButton.classList.add("is-link");
		phraseButton.classList.add("is-light");
		phraseButton.classList.remove("is-info");
		alterTokenPanel.style.display = "block";
		tokenButton.classList.remove("is-link");
		tokenButton.classList.remove("is-light");
		tokenButton.classList.add("is-info");
	});
	// endregion

	// Hide alternatives panel initially
	ClearAlternatives();
});
