import pandas as pd
import itertools

# --- ALGORITMA APRIORI MANUAL ---
# Langkah 1 & 3: Menghitung Nilai Support
# Rumus: Support(A) = Count(A) / N
def calculate_support(itemset, transactions):
    count = 0
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    N = len(transactions)
    return count / N

# Langkah 2: Membentuk Kandidat Itemset (Ck)
# Rumus: Ck = Fk-1 JOIN Fk-1
def generate_candidates(prev_frequent_itemsets, k):
    candidates = set()
    # Menggabungkan itemset (Self-Join)
    for itemset1 in prev_frequent_itemsets:
        for itemset2 in prev_frequent_itemsets:
            # Gabungkan dua itemset
            union_set = itemset1.union(itemset2)
            # Kandidat dengan panjang k
            if len(union_set) == k:
                candidates.add(union_set)
    return list(candidates)

# Langkah 4: Melakukan Pruning (Apriori Property)
def pruning(candidates, prev_frequent_itemsets, k):
    pruned_candidates = []
    for candidate in candidates:
        is_valid = True
        # Cek semua subset dari kandidat
        subsets = itertools.combinations(candidate, k-1)
        for subset in subsets:
            # Jika ada subset yang TIDAK ditemukan di itemset frequent sebelumnya
            if frozenset(subset) not in prev_frequent_itemsets:
                is_valid = False # Prune (pangkas) kandidat ini
                break
        if is_valid:
            pruned_candidates.append(candidate)
    return pruned_candidates

# --- ALGORITMA UTAMA ---

def apriori_manual(transactions, min_support):
    trans_sets = [set(t) for t in transactions]
    
    # 1. Mencari Frequent Itemset level-1 (C1 -> L1)
    items = set()
    for t in trans_sets:
        items.update(t)
    
    frequent_itemsets = {} # Menyimpan semua itemset yang lolos
    level_frequent = [] # L1, L2, dst
    
    # Hitung support untuk 1 item
    current_l = []
    for item in items:
        itemset = frozenset([item])
        supp = calculate_support(itemset, trans_sets)
        if supp >= min_support:
            current_l.append(itemset)
            frequent_itemsets[itemset] = supp
            
    level_frequent.append(set(current_l))

    k = 2
    while len(current_l) > 0:
        # Langkah 2: Generate Candidate Ck
        candidates = generate_candidates(current_l, k)
        # Langkah 4: Pruning
        candidates_pruned = pruning(candidates, level_frequent[-1], k)
        # Langkah 3: Hitung Support Kandidat yang tersisa
        next_l = []
        for candidate in candidates_pruned:
            supp = calculate_support(candidate, trans_sets)
            # Filter Support >= MinSupport
            if supp >= min_support:
                next_l.append(candidate)
                frequent_itemsets[candidate] = supp
        
        if not next_l:
            break
            
        level_frequent.append(set(next_l))
        current_l = next_l
        k += 1
        
    return frequent_itemsets

# Langkah 5: Menghasilkan Aturan Asosiasi (Confidence & Lift)
def generate_rules(frequent_itemsets, min_confidence, transactions_len):
    rules = []
    seen_unordered = set()  # untuk mencegah duplikat pasangan (A,B) dan (B,A)
    for itemset, support_ab in frequent_itemsets.items():
        if len(itemset) > 1:
            # Buat semua kemungkinan kombinasi (A -> B)
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset.difference(antecedent)

                    # Skip jika pasangan ini (tanpa arah) sudah diproses
                    unordered_key = frozenset([antecedent, consequent])
                    if unordered_key in seen_unordered:
                        continue

                    # Pastikan antecedent dan consequent ada di frequent_itemsets
                    if antecedent not in frequent_itemsets or consequent not in frequent_itemsets:
                        continue

                    support_a = frequent_itemsets[antecedent]
                    support_b = frequent_itemsets[consequent]

                    # Rumus Confidence
                    confidence = support_ab / support_a

                    # Rumus Lift 
                    lift = support_ab / (support_a * support_b)

                    if confidence >= min_confidence:
                        rules.append({
                            'Antecedent': list(antecedent),
                            'Consequent': list(consequent),
                            'Support': round(support_ab, 4),
                            'Confidence': round(confidence, 4),
                            'Lift': round(lift, 4)
                        })
                        # Tandai pasangan ini (tanpa arah) sudah diproses
                        seen_unordered.add(unordered_key)
    return pd.DataFrame(rules)

# --- EKSEKUSI PROGRAM ---

# 1. Load Data 
try:
    full_dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
    print(f"Dataset dimuat: {full_dataset.shape[0]} transaksi, maksimal {full_dataset.shape[1]} item per transaksi.")
except FileNotFoundError:
    print("Error: File 'Market_Basket_Optimisation.csv' tidak ditemukan.")
    exit()

print("Sedang memproses data transaksi...")
transactions = []
num_columns = full_dataset.shape[1] 

for i in range(0, len(full_dataset)):
    transaction = []
    for j in range(0, num_columns):
        item = str(full_dataset.values[i, j])
        # Filter 'nan' (Pandas NaN) dan 'None'
        if item != 'nan' and item != 'None':
            transaction.append(item)
    transactions.append(transaction)

# 2. Jalankan Apriori
# Support 5% (0.05) dan Confidence 20% (0.2)
min_support = 0.01 
min_confidence = 0.2

print(f"\nMenjalankan Apriori (Min Support: {min_support}, Min Confidence: {min_confidence})...")
freq_items = apriori_manual(transactions, min_support=min_support)

# Cek jika freq_items kosong
if not freq_items:
    print(f"\n[!] Tidak ditemukan Frequent Itemset dengan support = {min_support}.")
else:
    rules_df = generate_rules(freq_items, min_confidence=min_confidence, transactions_len=len(transactions))

    # 3. Tampilkan Hasil
    print("\n====== Frequent Itemsets  (Top 5) ======")
    # Mengurutkan itemset berdasarkan support tertinggi
    sorted_items = sorted(freq_items.items(), key=lambda x: x[1], reverse=True)
    for item, support in sorted_items[:5]:
        print(f"Item: {list(item)}, Support: {support:.4f}")

    print(f"\n=========== Association Rules  (Top 10 by Lift) ===========")
    if not rules_df.empty:
        # Tampilkan format yang lebih rapi
        print(rules_df.sort_values(by='Lift', ascending=False).head(10).to_string(index=False))
    else:
        print("Tidak ada aturan asosiasi yang memenuhi minimum confidence.")