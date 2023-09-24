class InpaintingForensics():
    def __init__(self):
        self.train_num = 48000
        self.val_num = 1000
        self.test_num = 12
        self.batch_size = 24
        # For training, please provide the absolute path of training data that saved in numpy with following format
        # E.g., file = [['./training_input_1.png', './training_ground_truth_1.png'],
        #              ['./training_input_2.png', './training_ground_truth_2.png'],...]
        self.train_file = ''
        self.val_file = ''
        self.test_file = ''
        train_dataset = IID_Dataset(self.train_num, self.train_file, choice='train')
        val_dataset = IID_Dataset(self.val_num, self.val_file, choice='val')
        test_dataset = IID_Dataset(self.test_num, self.test_file, choice='test')

        self.giid_model = IID_Model().cuda()
        self.n_epochs = 1000
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

    def train(self):
        with open('log.txt', 'a+') as f:
            f.write('\nTrain %s with %d' % (self.train_file, self.train_num))
            f.write('\nVal %s with %d' % (self.val_file, self.val_num))
            f.write('\nTest %s with %d' % (self.test_file, self.test_num))
        scheduler_gan = ReduceLROnPlateau(self.giid_model.gen_optimizer, patience=10, factor=0.5)
        best_auc = 0
        for epoch in range(self.n_epochs):
            cnt, gen_losses, auc = 0, [], []
            for items in self.train_loader:
                cnt += self.batch_size
                self.giid_model.train()
                Ii, Mg = (item.cuda() for item in items[:-1])
                Mo, gen_loss = self.giid_model.process(Ii, Mg)
                self.giid_model.backward(gen_loss)
                gen_losses.append(gen_loss.item())
                Mg, Mo = self.convert2(Mg), self.convert2(Mo)
                N, H, W, C = Mg.shape
                auc.append(roc_auc_score(Mg.reshape(N * H * W * C).astype('int'), Mo.reshape(N * H * W * C)) * 100.)
                print('Tra (%d/%d): G:%6.3f A:%3.2f' % (cnt, self.train_num, np.mean(gen_losses), np.mean(auc)), end='\r')
                if cnt % 12000 == 0 or cnt >= self.train_num:
                    val_gen_loss, val_auc = self.val()
                    scheduler_gan.step(val_auc)
                    print('Val (%d/%d): G:%6.3f A:%3.2f' % (cnt, self.train_num, val_gen_loss, val_auc))
                    if val_auc > best_auc:
                        best_auc = val_auc
                        self.giid_model.save('best/')
                    self.giid_model.save('latest/')
                    with open('log.txt', 'a+') as f:
                        f.write('\n(%d/%d): Tra: A:%4.2f Val: A:%4.2f' % (cnt, self.train_num, np.mean(auc), val_auc))
                    auc, gen_losses = [], []

    def val(self):
        self.giid_model.eval()
        auc, gen_losses = [], []
        for cnt, items in enumerate(self.val_loader):
            Ii, Mg = (item.cuda() for item in items[:-1])
            filename = items[-1][0]
            Mo, gen_loss = self.giid_model.process(Ii, Mg)
            gen_losses.append(gen_loss.item())
            Ii, Mg, Mo = self.convert1(Ii), self.convert2(Mg)[0], self.convert2(Mo)[0]
            H, W, _ = Mg.shape
            auc.append(roc_auc_score(Mg.reshape(H * W).astype('int'), Mo.reshape(H * W)) * 100.)

            # Sample 100 validation images for visualization
            if len(auc) <= 100:
                Mg, Mo = Mg * 255, Mo * 255
                out = np.zeros([H, H * 3, 3])
                out[:, :H, :] = Ii
                out[:, H:H*2, :] = np.concatenate([Mo, Mo, Mo], axis=2)
                out[:, H*2:, :] = np.concatenate([Mg, Mg, Mg], axis=2)
                cv2.imwrite('demo_val/val_' + filename, out)
        return np.mean(gen_losses), np.mean(auc)

    def test(self):
        self.giid_model.load()
        self.giid_model.eval()
        for cnt, items in enumerate(self.test_loader):
            print(cnt, end='\r')
            Ii, Mg = (item.cuda() for item in items[:-1])
            filename = items[-1][0]
            Mo, gen_loss = self.giid_model.process(Ii, Mg)
            Ii, Mo = self.convert1(Ii), self.convert2(Mo)[0]
            cv2.imwrite('demo_output/output_' + filename, Mo * 255)

    def convert1(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
        return img

    def convert2(self, x):
        return x.permute(0, 2, 3, 1).cpu().detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='train or test the model', choices=['train', 'test'])
    args = parser.parse_args()

    model = InpaintingForensics()
    if args.type == 'train':
        model.train()
    elif args.type == 'test':
        model.test()