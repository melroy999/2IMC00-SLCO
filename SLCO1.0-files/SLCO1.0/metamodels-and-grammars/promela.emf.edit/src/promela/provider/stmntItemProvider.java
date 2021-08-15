/**
 */
package promela.provider;


import java.util.Collection;
import java.util.List;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.ComposeableAdapterFactory;
import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.ViewerNotification;

import promela.PromelaFactory;
import promela.PromelaPackage;
import promela.stmnt;

/**
 * This is the item provider adapter for a {@link promela.stmnt} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class stmntItemProvider
	extends stepItemProvider
	implements
		IEditingDomainItemProvider,
		IStructuredItemContentProvider,
		ITreeItemContentProvider,
		IItemLabelProvider,
		IItemPropertySource {
	/**
	 * This constructs an instance from a factory and a notifier.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public stmntItemProvider(AdapterFactory adapterFactory) {
		super(adapterFactory);
	}

	/**
	 * This returns the property descriptors for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public List<IItemPropertyDescriptor> getPropertyDescriptors(Object object) {
		if (itemPropertyDescriptors == null) {
			super.getPropertyDescriptors(object);

			addLabelPropertyDescriptor(object);
		}
		return itemPropertyDescriptors;
	}

	/**
	 * This adds a property descriptor for the Label feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addLabelPropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_stmnt_label_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_stmnt_label_feature", "_UI_stmnt_type"),
				 PromelaPackage.Literals.STMNT__LABEL,
				 true,
				 false,
				 false,
				 ItemPropertyDescriptor.GENERIC_VALUE_IMAGE,
				 null,
				 null));
	}

	/**
	 * This specifies how to implement {@link #getChildren} and is used to deduce an appropriate feature for an
	 * {@link org.eclipse.emf.edit.command.AddCommand}, {@link org.eclipse.emf.edit.command.RemoveCommand} or
	 * {@link org.eclipse.emf.edit.command.MoveCommand} in {@link #createCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Collection<? extends EStructuralFeature> getChildrenFeatures(Object object) {
		if (childrenFeatures == null) {
			super.getChildrenFeatures(object);
			childrenFeatures.add(PromelaPackage.Literals.STMNT__UNLESS);
		}
		return childrenFeatures;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EStructuralFeature getChildFeature(Object object, Object child) {
		// Check the type of the specified child object and return the proper feature to use for
		// adding (see {@link AddCommand}) it as a child.

		return super.getChildFeature(object, child);
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		String label = ((stmnt)object).getLabel();
		return label == null || label.length() == 0 ?
			getString("_UI_stmnt_type") :
			getString("_UI_stmnt_type") + " " + label;
	}

	/**
	 * This handles model notifications by calling {@link #updateChildren} to update any cached
	 * children and by creating a viewer notification, which it passes to {@link #fireNotifyChanged}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void notifyChanged(Notification notification) {
		updateChildren(notification);

		switch (notification.getFeatureID(stmnt.class)) {
			case PromelaPackage.STMNT__LABEL:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), false, true));
				return;
			case PromelaPackage.STMNT__UNLESS:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), true, false));
				return;
		}
		super.notifyChanged(notification);
	}

	/**
	 * This adds {@link org.eclipse.emf.edit.command.CommandParameter}s describing the children
	 * that can be created under this object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected void collectNewChildDescriptors(Collection<Object> newChildDescriptors, Object object) {
		super.collectNewChildDescriptors(newChildDescriptors, object);

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createvarref()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createsend()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createreceive()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createassign_std()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createassign_inc()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createassign_dec()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createif_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createdo_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createatomic_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.created_step_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createblock_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createelse_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createbreak_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.creategoto_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createprint_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createassert_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createc_code()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createc_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createc_decl()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createc_track()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createc_state()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createbin_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createcond_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createlen_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createtimeout_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createnp__expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createenabled_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createpc_value_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createname_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createrun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createandor_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createchanpoll_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createnum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createenum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.STMNT__UNLESS,
				 PromelaFactory.eINSTANCE.createmtype_const()));
	}

}
